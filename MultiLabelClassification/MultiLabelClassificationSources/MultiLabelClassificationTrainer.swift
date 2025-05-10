import CoreML
import CreateML
import CreateMLComponents
import Foundation
import SCSInterface

public class MultiLabelClassificationTrainer: ScreeningTrainerProtocol {
    public typealias TrainingResultType = MultiLabelTrainingResult

    struct ManifestEntry: Decodable {
        var filename: String
        var annotations: [String]
    }

    public var modelName: String { "ScaryCatScreeningML_MultiLabel" }
    public var customOutputDirPath: String { "MultiLabelClassification/OutputModels" }

    public var manifestFileName: String { "multilabel_cat_annotations.json" }

    public var resourcesDirectoryPath: String {
        var dir = URL(fileURLWithPath: #filePath)
        dir.deleteLastPathComponent()
        dir.deleteLastPathComponent()
        return dir.appending(path: "Resources").path
    }

    public init() {}

    public func train(author: String, shortDescription _: String, version: String) async -> MultiLabelTrainingResult? {
        do {
            let resourcesDir = URL(fileURLWithPath: resourcesDirectoryPath)
            let manifestURL = resourcesDir.appendingPathComponent(manifestFileName)

            guard let jsonData = try? Data(contentsOf: manifestURL),
                  let entries = try? JSONDecoder().decode([ManifestEntry].self, from: jsonData)
            else {
                print("❌ Error: Failed to load or parse manifest file at \(manifestURL.path)")
                return nil
            }

            let annotatedFeatures = entries.map {
                AnnotatedFeature(
                    feature: resourcesDir.appendingPathComponent($0.filename),
                    annotation: Set($0.annotations)
                )
            }

            let labels = Set(annotatedFeatures.flatMap(\.annotation)).sorted()
            guard !labels.isEmpty else {
                print("❌ Error: No labels found in the training data.")
                return nil
            }

            let pipeline = ImageReader()
                .appending(ImageFeaturePrint(revision: 1))
                .appending(FullyConnectedNetworkMultiLabelClassifier<Float, String>(labels: Set(labels)))

            let (train, val) = annotatedFeatures.randomSplit(by: 0.8)
            let start = Date()
            let model = try await pipeline.fitted(to: train, validateOn: val)
            let duration = Date().timeIntervalSince(start)

            var projectRootURL = URL(fileURLWithPath: #filePath)
            projectRootURL
                .deleteLastPathComponent() // .../MultiLabelClassificationSources/MultiLabelClassificationTrainer.swift
            projectRootURL.deleteLastPathComponent() // .../MultiLabelClassificationSources/
            projectRootURL.deleteLastPathComponent() // .../MultiLabelClassification/
            projectRootURL.deleteLastPathComponent() // プロジェクトルートへ

            // Base output directory (e.g., MultiLabelClassification/OutputModels)
            let baseOutputDirURL = projectRootURL.appendingPathComponent(customOutputDirPath)

            // Create the version-specific directory (e.g., MultiLabelClassification/OutputModels/v1)
            let versionedOutputDirURL = baseOutputDirURL.appendingPathComponent(version)
            try FileManager.default.createDirectory(
                at: versionedOutputDirURL,
                withIntermediateDirectories: true,
                attributes: nil
            )
            print("📂 Versioned output directory: \(versionedOutputDirURL.path)")

            // List existing runs within the version-specific directory
            let existingRuns = (try? FileManager.default.contentsOfDirectory(at: versionedOutputDirURL, includingPropertiesForKeys: nil)) ?? []
            
            // Define the prefix for run names, including the version
            let runNamePrefix = "MultiLabel_\(version)_Result_"
            
            // Calculate the next run index
            let nextIndex = (existingRuns.compactMap { url -> Int? in
                let runName = url.lastPathComponent
                if runName.hasPrefix(runNamePrefix) {
                    return Int(runName.replacingOccurrences(of: runNamePrefix, with: ""))
                }
                return nil
            }.max() ?? 0) + 1
            
            // Construct the main output run URL with the version in its name
            let finalOutputDir = versionedOutputDirURL.appendingPathComponent("\(runNamePrefix)\(nextIndex)")
            
            try FileManager.default.createDirectory(
                at: finalOutputDir,
                withIntermediateDirectories: false, // This should be true if a parent might not exist, but versionedOutputDirURL creation handles it
                attributes: nil
            )
            print("💾 Result directory: \(finalOutputDir.path)")

            let modelURL = finalOutputDir.appendingPathComponent("\(modelName)_\(version).mlmodel")

            let metadata = ModelMetadata(version: version, author: author)
            try model.export(to: modelURL, metadata: metadata)

            let predictions = try await model.prediction(from: val)
            let metrics = try MultiLabelClassificationMetrics(
                classifications: predictions.map(\.prediction),
                groundTruth: predictions.map(\.annotation),
                strategy: .balancedPrecisionAndRecall,
                labels: Set(labels)
            )

            return MultiLabelTrainingResult(
                modelName: modelName,
                trainingDataAccuracy: 0.0,
                validationDataAccuracy: Double(metrics.meanAveragePrecision),
                trainingDataError: 0.0,
                validationDataError: 0.0,
                trainingDuration: duration,
                modelOutputPath: modelURL.path,
                trainingDataPath: manifestURL.path,
                classLabels: labels
            )
        } catch let error as CreateML.MLCreateError {
            print("  ❌ Model [\(modelName)] training or saving error (CreateML): \(error.localizedDescription)")
            return nil
        } catch {
            print("  ❌ An unexpected error occurred during the training process: \(error.localizedDescription)")
            if let nsError = error as NSError? {
                print("    Detailed error information: \(nsError.userInfo)")
            }
            return nil
        }
    }
}

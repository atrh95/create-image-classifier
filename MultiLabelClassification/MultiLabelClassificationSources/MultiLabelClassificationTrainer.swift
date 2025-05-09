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

    public func train(author: String, shortDescription: String, version: String) async -> MultiLabelTrainingResult? {
        do {
            let resourcesDir = URL(fileURLWithPath: resourcesDirectoryPath)
            let manifestURL = resourcesDir.appendingPathComponent(manifestFileName)

            guard let jsonData = try? Data(contentsOf: manifestURL),
                  let entries = try? JSONDecoder().decode([ManifestEntry].self, from: jsonData) else {
                print("❌ Error: Failed to load or parse manifest file at \(manifestURL.path)")
                return nil
            }

            let annotatedFeatures = entries.map {
                AnnotatedFeature(
                    feature: resourcesDir.appendingPathComponent($0.filename),
                    annotation: Set($0.annotations)
                )
            }

            let labels = Set(annotatedFeatures.flatMap { $0.annotation }).sorted()
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
            projectRootURL.deleteLastPathComponent() // MultiLabelClassificationSources
            projectRootURL.deleteLastPathComponent() // MultiLabelClassification
            projectRootURL.deleteLastPathComponent() // プロジェクトルート

            let baseTargetOutputDir = projectRootURL.appendingPathComponent(customOutputDirPath)
            try FileManager.default.createDirectory(at: baseTargetOutputDir, withIntermediateDirectories: true, attributes: nil)
            print("📂 Base output directory: \(baseTargetOutputDir.path)")

            var resultCounter = 1
            let resultDirPrefix = "multilabel_result_"
            var finalOutputDir: URL!
            repeat {
                let resultDirName = "\(resultDirPrefix)\(resultCounter)"
                finalOutputDir = baseTargetOutputDir.appendingPathComponent(resultDirName)
                resultCounter += 1
            } while FileManager.default.fileExists(atPath: finalOutputDir.path)
            try FileManager.default.createDirectory(at: finalOutputDir, withIntermediateDirectories: false, attributes: nil)
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
                trainingAccuracy: 0.0,
                validationAccuracy: Double(metrics.meanAveragePrecision),
                trainingError: 0.0,
                validationError: 0.0,
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

//
//  MultiLabelClassificationTrainer.swift
//
//  Re‑written from scratch on 2025‑05‑15.
//  Provides a robust, fully‑typed multi‑label training pipeline
//  using CreateMLComponents.
//

import CoreML
import CreateML
import CreateMLComponents
import CSInterface
import Foundation

/// Trains and exports a multi‑label image classification model for the
/// CatScreeningML tool‑chain.
///
/// The trainer expects a manifest JSON file of the form:
/// ```json
/// [
///   { "filename": "cat_001.jpg", "annotations": ["scary", "openMouth"] },
///   { "filename": "cat_002.jpg", "annotations": ["cute"] }
/// ]
/// ```
public final class MultiLabelClassificationTrainer: ScreeningTrainerProtocol {
    // MARK: - Types

    public typealias TrainingResultType = MultiLabelTrainingResult

    private struct ManifestEntry: Decodable {
        let filename: String
        let annotations: [String]
    }

    // MARK: - Configuration

    public var modelName: String { "ScaryCatScreeningML_MultiLabel" }
    public var customOutputDirPath: String { "MultiLabelClassification/OutputModels" }
    public var outputRunNamePrefix: String { "MultiLabel" }
    public var manifestFileName: String { "multilabel_cat_annotations.json" }

    /// Directory that contains training resources (manifest + images).
    public var resourcesDirectoryPath: String {
        var dir = URL(fileURLWithPath: #filePath)
        dir.deleteLastPathComponent() // .../MultiLabelClassificationSources
        dir.deleteLastPathComponent() // .../MultiLabelClassification
        return dir.appending(path: "Resources").path
    }

    /// Confidence threshold for turning a soft distribution into hard labels.
    private let predictionThreshold: Float = 0.5

    // MARK: - Init

    public init() {}

    // MARK: - Public API

    public func train(
        author _: String,
        version: String,
        maxIterations: Int
    ) async -> MultiLabelTrainingResult? {
        // 1) Prepare output directory
        let outputDir: URL
        do {
            outputDir = try setupVersionedRunOutputDirectory(
                version: version,
                trainerFilePath: #filePath
            )
        } catch {
            print("❌ Failed to create output directory – \(error.localizedDescription)")
            return nil
        }

        // 2) Load manifest entries
        let resourcesDir = URL(fileURLWithPath: resourcesDirectoryPath)
        let manifestURL = resourcesDir.appending(path: manifestFileName)

        guard
            let manifestData = try? Data(contentsOf: manifestURL),
            let entries = try? JSONDecoder().decode([ManifestEntry].self, from: manifestData),
            !entries.isEmpty
        else {
            print("❌ Could not read or decode manifest at \(manifestURL.path)")
            return nil
        }

        // 3) Build AnnotatedFeature array
        let annotatedFeatures: [AnnotatedFeature<URL, Set<String>>] = entries.compactMap { entry in
            let fileURL = resourcesDir.appending(path: entry.filename)
            return AnnotatedFeature(feature: fileURL, annotation: Set(entry.annotations))
        }

        // 4) Establish full label set
        let labels = Set(annotatedFeatures.flatMap(\.annotation)).sorted()
        guard !labels.isEmpty else {
            print("❌ No labels detected in manifest.")
            return nil
        }
        print("📚 Labels: \(labels.joined(separator: ", "))")

        // 5) Build CreateMLComponents pipeline
        let classifier = FullyConnectedNetworkMultiLabelClassifier<Float, String>(
            labels: Set(labels)
        )
        let featureExtractor = ImageFeaturePrint(revision: 1)
        let pipeline = featureExtractor.appending(classifier)

        // 6) Train / validate
        let reader = ImageReader()
        let (trainSet, validationSet) = annotatedFeatures.randomSplit(by: 0.8)
        guard
            let trainingFeatures = try? await reader.applied(to: trainSet),
            let validationFeatures = try? await reader.applied(to: validationSet)
        else {
            print("❌ Failed to apply image reader")
            return nil
        }

        print("⏳ Training – train: \(trainSet.count) / validation: \(validationSet.count)")

        let t0 = Date()
        let fittedPipeline: ComposedTransformer<
            ImageFeaturePrint,
            FullyConnectedNetworkMultiLabelClassifier<Float, String>.Transformer
        >
        do {
            fittedPipeline = try await pipeline.fitted(to: trainingFeatures, validateOn: validationFeatures)
        } catch {
            print("❌ Training failed – \(error.localizedDescription)")
            return nil
        }
        let trainingTime = Date().timeIntervalSince(t0)
        print("🎉 Training complete in \(String(format: "%.2f", trainingTime)) s")

        // 7) Export .mlmodel
        let modelURL = outputDir.appendingPathComponent("\(modelName)_\(version).mlmodel")
        do {
            try fittedPipeline.export(to: modelURL)
            print("✅ Saved model to \(modelURL.path)")
        } catch {
            print("❌ Failed to export model – \(error.localizedDescription)")
            return nil
        }

        // 8) Evaluate mAP + per‑label metrics
        // Evaluation skipped in this minimal compile‑fix pass
        let meanAP: Double? = nil
        let perLabelSummary = "evaluation skipped"
        let avgRecall: Double? = nil
        let avgPrecision: Double? = nil

        // 9) Assemble result
        return MultiLabelTrainingResult(
            modelName: modelName,
            trainingDurationInSeconds: trainingTime,
            modelOutputPath: modelURL.path,
            trainingDataPath: manifestURL.path,
            classLabels: labels,
            maxIterations: maxIterations,
            meanAveragePrecision: meanAP,
            perLabelMetricsSummary: perLabelSummary,
            averageRecallAcrossLabels: avgRecall,
            averagePrecisionAcrossLabels: avgPrecision
        )
    }
}

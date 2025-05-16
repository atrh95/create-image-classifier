import CoreML
import CreateML
import CreateMLComponents
import CSInterface
import Foundation

public final class MultiLabelClassificationTrainer: ScreeningTrainerProtocol {
    public typealias TrainingResultType = MultiLabelTrainingResult

    private struct ManifestEntry: Decodable {
        let filename: String
        let annotations: [String]
    }

    public var outputDirPath: String {
        var dir = URL(fileURLWithPath: #filePath)
        dir.deleteLastPathComponent()
        dir.deleteLastPathComponent()
        return dir.appendingPathComponent("OutputModels").path
    }

    public var classificationMethod: String { "MultiLabel" }
    public var manifestFileName: String { "multilabel_cat_annotations.json" }

    public var resourcesDirectoryPath: String {
        var dir = URL(fileURLWithPath: #filePath)
        dir.deleteLastPathComponent()
        dir.deleteLastPathComponent()
        return dir.appending(path: "Resources").path
    }

    /// ソフトな分布をハードラベルに変換するための信頼度の閾値
    private let predictionThreshold: Float = 0.5

    public init() {}

    public func train(
        author _: String,
        modelName: String,
        version: String,
        maxIterations: Int
    ) async -> MultiLabelTrainingResult? {
        let outputDir: URL
        do {
            outputDir = try createOutputDirectory(
                modelName: modelName,
                version: version
            )
        } catch {
            print("❌ Failed to create output directory – \(error.localizedDescription)")
            return nil
        }

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

        let annotatedFeatures: [AnnotatedFeature<URL, Set<String>>] = entries.compactMap { entry in
            let fileURL = resourcesDir.appending(path: entry.filename)
            return AnnotatedFeature(feature: fileURL, annotation: Set(entry.annotations))
        }

        let labels = Set(annotatedFeatures.flatMap(\.annotation)).sorted()
        guard !labels.isEmpty else {
            print("❌ No labels detected in manifest.")
            return nil
        }
        print("📚 Labels: \(labels.joined(separator: ", "))")

        let classifier = FullyConnectedNetworkMultiLabelClassifier<Float, String>(
            labels: Set(labels)
        )
        let featureExtractor = ImageFeaturePrint(revision: 1)
        let pipeline = featureExtractor.appending(classifier)

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

        let modelURL = outputDir.appendingPathComponent("\(modelName)_\(classificationMethod)_\(version).mlmodel")
        do {
            try fittedPipeline.export(to: modelURL)
            print("✅ Saved model to \(modelURL.path)")
        } catch {
            print("❌ Failed to export model – \(error.localizedDescription)")
            return nil
        }

        let meanAP: Double? = nil
        let perLabelSummary = "evaluation skipped"
        let avgRecall: Double? = nil
        let avgPrecision: Double? = nil

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

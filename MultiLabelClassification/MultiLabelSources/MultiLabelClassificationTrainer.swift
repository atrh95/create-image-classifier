import CoreML
import CreateML
import CreateMLComponents
import CICConfusionMatrix
import CICInterface
import Foundation

public final class MultiLabelClassificationTrainer: ScreeningTrainerProtocol {
    public typealias TrainingResultType = MultiLabelTrainingResult

    public struct ManifestEntry: Decodable {
        let filename: String
        let annotations: [String]
    }

    private let resourcesDirectoryPathOverride: String?
    private let outputDirectoryPathOverride: String?
    private let annotationFileNameOverride: String?

    public var outputDirPath: String {
        if let overridePath = outputDirectoryPathOverride {
            return overridePath
        }
        var dir = URL(fileURLWithPath: #filePath)
        dir.deleteLastPathComponent()
        dir.deleteLastPathComponent()
        return dir.appendingPathComponent("OutputModels").path
    }

    public var classificationMethod: String { "MultiLabel" }

    public var resourcesDirectoryPath: String {
        if let overridePath = resourcesDirectoryPathOverride {
            return overridePath
        }
        var dir = URL(fileURLWithPath: #filePath)
        dir.deleteLastPathComponent()
        dir.deleteLastPathComponent()
        return dir.appending(path: "Resources").path
    }

    // ラベル判定の信頼度閾値
    private let predictionThreshold: Float = 0.5

    public init(
        resourcesDirectoryPathOverride: String? = nil,
        outputDirectoryPathOverride: String? = nil,
        annotationFileNameOverride: String? = nil
    ) {
        self.resourcesDirectoryPathOverride = resourcesDirectoryPathOverride
        self.outputDirectoryPathOverride = outputDirectoryPathOverride
        self.annotationFileNameOverride = annotationFileNameOverride
    }

    public func train(
        author: String,
        modelName: String,
        version: String,
        modelParameters: CreateML.MLImageClassifier.ModelParameters,
        scenePrintRevision: Int?
    ) async -> MultiLabelTrainingResult? {
        let outputDir: URL
        do {
            outputDir = try createOutputDirectory(
                modelName: modelName,
                version: version
            )
        } catch {
            print("🛑 エラー: 出力ディレクトリの作成に失敗しました – \(error.localizedDescription)")
            return nil
        }

        let resourcesDir = URL(fileURLWithPath: resourcesDirectoryPath)

        let currentAnnotationFileName: String
        if let overrideName = annotationFileNameOverride {
            currentAnnotationFileName = overrideName
        } else {
            let fileManager = FileManager.default
            do {
                let items = try fileManager.contentsOfDirectory(
                    at: resourcesDir,
                    includingPropertiesForKeys: nil,
                    options: .skipsHiddenFiles
                )
                if let jsonFile = items.first(where: { $0.pathExtension.lowercased() == "json" }) {
                    currentAnnotationFileName = jsonFile.lastPathComponent
                } else {
                    print("🛑 トレーニングエラー: リソースディレクトリ「\(resourcesDirectoryPath)」でJSONアノテーションファイルが見つかりませんでした。(オーバーライドも未指定)")
                    return nil
                }
            } catch {
                print(
                    "🛑 エラー: リソースディレクトリ「\(resourcesDirectoryPath)」の内容読み取り中にエラーが発生しました: \(error.localizedDescription)"
                )
                return nil
            }
        }

        let annotationFileURL = resourcesDir.appending(path: currentAnnotationFileName)

        guard FileManager.default.fileExists(atPath: annotationFileURL.path) else {
            print("🛑 エラー: アノテーションファイルが見つかりません: \(annotationFileURL.path)")
            return nil
        }

        guard
            let manifestData = try? Data(contentsOf: annotationFileURL),
            let entries = try? JSONDecoder().decode([ManifestEntry].self, from: manifestData),
            !entries.isEmpty
        else {
            print("🛑 エラー: アノテーションファイルの読み取りまたはデコードに失敗しました: \(annotationFileURL.path)")
            return nil
        }

        let annotatedFeatures: [AnnotatedFeature<URL, Set<String>>] = entries.compactMap { entry in
            let fileURL = resourcesDir.appending(path: entry.filename)
            return AnnotatedFeature(feature: fileURL, annotation: Set(entry.annotations))
        }

        let labels = Set(annotatedFeatures.flatMap(\.annotation)).sorted()
        guard !labels.isEmpty else {
            print("🛑 エラー: アノテーションファイルでラベルが検出されませんでした。")
            return nil
        }

        let classifier = FullyConnectedNetworkMultiLabelClassifier<Float, String>(
            labels: Set(labels)
        )
        let featureExtractor = ImageFeaturePrint(revision: scenePrintRevision ?? 1)
        let pipeline = featureExtractor.appending(classifier)

        let reader = ImageReader()
        let (trainSet, validationSet) = annotatedFeatures.randomSplit(by: 0.8)

        guard
            let trainingFeatures = try? await reader.applied(to: trainSet),
            let validationFeatures = try? await reader.applied(to: validationSet)
        else {
            print("🛑 エラー: 画像リーダーの適用に失敗しました。学習データまたは検証データの処理中にエラーが発生しました。")
            return nil
        }

        let t0 = Date()
        let fittedPipeline: ComposedTransformer<
            ImageFeaturePrint,
            FullyConnectedNetworkMultiLabelClassifier<Float, String>.Transformer
        >
        do {
            fittedPipeline = try await pipeline.fitted(to: trainingFeatures, validateOn: validationFeatures)
        } catch {
            print("🛑 エラー: トレーニングに失敗しました – \(error.localizedDescription)")
            return nil
        }
        let trainingTime = Date().timeIntervalSince(t0)

        // 評価メトリクスを直接取得せず、混同行列に基づいて算出
        let trainingError = 1.0 // 評価指標は未算出のため仮値
        let validationError: Double = await {
            guard let validationPredictions = try? await fittedPipeline.applied(to: validationFeatures) else {
                return 1.0
            }

            var predictions: [(trueLabels: Set<String>, predictedLabels: Set<String>)] = []
            for i in 0 ..< validationSet.count {
                let trueAnnotations = validationSet[i].annotation
                let actualDistribution = validationPredictions[i].feature

                var predictedLabels = Set<String>()
                for labelInDataset in labels {
                    if let score = actualDistribution[labelInDataset], score >= predictionThreshold {
                        predictedLabels.insert(labelInDataset)
                    }
                }

                predictions.append((trueLabels: trueAnnotations, predictedLabels: predictedLabels))
            }

            let confusionMatrix = CSMultiLabelConfusionMatrix(
                predictions: predictions,
                labels: labels,
                predictionThreshold: predictionThreshold
            )

            // F1スコアの平均に基づいて簡易的なエラー率を推定（仮）
            let metrics = confusionMatrix.calculateMetrics()
            let avgF1 = metrics.compactMap(\.f1Score).reduce(0, +) / Double(metrics.count)
            let avgRecall = metrics.compactMap(\.recall).reduce(0, +) / Double(metrics.count)
            return 1.0 - (avgF1 + avgRecall) / 2.0
        }()

        var predictions: [(trueLabels: Set<String>, predictedLabels: Set<String>)] = []
        if let validationPredictions = try? await fittedPipeline.applied(to: validationFeatures) {
            for i in 0 ..< validationSet.count {
                let trueAnnotations = validationSet[i].annotation
                let annotatedPrediction = validationPredictions[i]
                let actualDistribution = annotatedPrediction.feature

                var predictedLabels = Set<String>()
                for labelInDataset in labels {
                    if let score = actualDistribution[labelInDataset], score >= predictionThreshold {
                        predictedLabels.insert(labelInDataset)
                    }
                }

                predictions.append((trueLabels: trueAnnotations, predictedLabels: predictedLabels))
            }
        }

        // 混同行列の計算をCSMultiLabelConfusionMatrixに委任
        let confusionMatrix = CSMultiLabelConfusionMatrix(
            predictions: predictions,
            labels: labels,
            predictionThreshold: predictionThreshold
        )

        var descriptionParts: [String] = []

        if !labels.isEmpty {
            descriptionParts.append("ラベル: " + labels.joined(separator: ", "))
        } else {
            descriptionParts.append("ラベル情報なし")
        }

        descriptionParts.append("最大反復回数 (指定値): \(modelParameters.maxIterations)回")
        descriptionParts.append(String(
            format: "学習データ数: %d枚, 検証データ数: %d枚",
            trainingFeatures.count,
            validationFeatures.count
        ))

        let metrics = confusionMatrix.calculateMetrics()
        if !metrics.isEmpty {
            descriptionParts.append("ラベル別検証指標 (しきい値: \(predictionThreshold)):")
            for metric in metrics {
                let metricsString = String(
                    format: "    %@: 再現率 %@, 適合率 %@, F1スコア %@",
                    metric.label,
                    metric.recall.map { String(format: "%.1f%%", $0 * 100) } ?? "計算不可",
                    metric.precision.map { String(format: "%.1f%%", $0 * 100) } ?? "計算不可",
                    metric.f1Score.map { String(format: "%.1f%%", $0 * 100) } ?? "計算不可"
                )
                descriptionParts.append(metricsString)
            }
        } else {
            descriptionParts.append("ラベル別検証指標: 計算スキップまたは失敗")
        }

        // データ拡張 (Data Augmentation)
        let augmentationFinalDescription: String
        if !modelParameters.augmentationOptions.isEmpty {
            augmentationFinalDescription = String(describing: modelParameters.augmentationOptions)
            descriptionParts.append("データ拡張: \(augmentationFinalDescription)")
        } else {
            augmentationFinalDescription = "なし"
            descriptionParts.append("データ拡張: なし")
        }

        // 特徴抽出器 (Feature Extractor)
        let featureExtractorTypeDescription = "ImageFeaturePrint"
        let featureExtractorDescForMetadata = if let revision = scenePrintRevision {
            "\(featureExtractorTypeDescription)(revision: \(revision))"
        } else {
            featureExtractorTypeDescription
        }
        descriptionParts.append("特徴抽出器: \(featureExtractorDescForMetadata)")

        let modelMetadata = ModelMetadata(
            description: """
            ラベル: \(labels.joined(separator: ", "))
            訓練正解率: \(String(format: "%.1f%%", (1.0 - trainingError) * 100.0))
            検証正解率: \(String(format: "%.1f%%", (1.0 - validationError) * 100.0))
            \(confusionMatrix.calculateMetrics().map { metric in
                """
                【\(metric.label)】
                再現率: \(metric.recall.map { String(format: "%.1f%%", $0 * 100.0) } ?? "計算不可"), \
                適合率: \(metric.precision.map { String(format: "%.1f%%", $0 * 100.0) } ?? "計算不可"), \
                F1スコア: \(metric.f1Score.map { String(format: "%.1f%%", $0 * 100.0) } ?? "計算不可")
                """
            }.joined(separator: "\n"))
            データ拡張: \(augmentationFinalDescription)
            特徴抽出器: \(featureExtractorDescForMetadata)
            """,
            version: version,
            author: author
        )

        let modelURL = outputDir.appendingPathComponent("\(modelName)_\(classificationMethod)_\(version).mlmodel")
        do {
            try fittedPipeline.export(to: modelURL, metadata: modelMetadata)
            print("✅ モデルを \(modelURL.path) に保存しました")
        } catch {
            print("🛑 エラー: モデルのエクスポートに失敗しました – \(error.localizedDescription)")
            return nil
        }

        return MultiLabelTrainingResult(
            modelName: modelName,
            trainingDurationInSeconds: trainingTime,
            modelOutputPath: modelURL.path,
            trainingDataPath: annotationFileURL.path,
            classLabels: labels,
            maxIterations: modelParameters.maxIterations,
            trainingMetrics: (
                accuracy: 1.0 - trainingError,
                errorRate: trainingError
            ),
            validationMetrics: (
                accuracy: 1.0 - validationError,
                errorRate: validationError
            ),
            dataAugmentationDescription: augmentationFinalDescription,
            featureExtractorDescription: featureExtractorTypeDescription,
            scenePrintRevision: scenePrintRevision,
            confusionMatrix: confusionMatrix
        )
    }
}

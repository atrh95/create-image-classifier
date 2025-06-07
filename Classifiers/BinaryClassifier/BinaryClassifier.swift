import CICConfusionMatrix
import CICFileManager
import CICInterface
import CICTrainingResult
import CoreML
import CreateML
import Foundation

public final class BinaryClassifier: ClassifierProtocol {
    public typealias TrainingResultType = BinaryTrainingResult

    private let fileManager: CICFileManager
    public var outputDirectoryPathOverride: String?
    public var resourceDirPathOverride: String?
    private var classImageCounts: [String: Int] = [:]

    public var outputParentDirPath: String {
        if let override = outputDirectoryPathOverride {
            return override
        }
        let currentFileURL = URL(fileURLWithPath: #filePath)
        return currentFileURL
            .deletingLastPathComponent() // BinaryClassifier
            .deletingLastPathComponent() // Sources
            .deletingLastPathComponent() // Classifiers
            .deletingLastPathComponent() // Project root
            .appendingPathComponent("CICOutputModels")
            .appendingPathComponent("BinaryClassifier")
            .path
    }

    public var resourcesDirectoryPath: String {
        if let override = resourceDirPathOverride {
            return override
        }
        let currentFileURL = URL(fileURLWithPath: #filePath)
        return currentFileURL
            .deletingLastPathComponent() // BinaryClassifier
            .deletingLastPathComponent() // Sources
            .deletingLastPathComponent() // Classifiers
            .deletingLastPathComponent() // Project root
            .appendingPathComponent("CICResources")
            .appendingPathComponent("BinaryResources")
            .path
    }

    public var classificationMethod: String { "Binary" }

    public init(
        outputDirectoryPathOverride: String? = nil,
        resourceDirPathOverride: String? = nil,
        fileManager: CICFileManager = CICFileManager()
    ) {
        self.outputDirectoryPathOverride = outputDirectoryPathOverride
        self.resourceDirPathOverride = resourceDirPathOverride
        self.fileManager = fileManager
    }

    public func create(
        author: String,
        modelName: String,
        version: String,
        modelParameters: CreateML.MLImageClassifier.ModelParameters
    ) async throws {
        print("📁 リソースディレクトリ: \(resourcesDirectoryPath)")
        print("🚀 2クラス分類モデル作成開始 (バージョン: \(version))...")

        // クラスラベルディレクトリの取得と検証
        let classLabelDirURLs = try fileManager.getClassLabelDirectories(resourcesPath: resourcesDirectoryPath)
        print("📁 検出されたクラスラベルディレクトリ: \(classLabelDirURLs.map(\.lastPathComponent).joined(separator: ", "))")

        guard classLabelDirURLs.count == 2 else {
            throw NSError(domain: "BinaryClassifier", code: -1, userInfo: [
                NSLocalizedDescriptionKey: "2クラス分類には2つのクラスラベルディレクトリが必要です。現在 \(classLabelDirURLs.count)個。",
            ])
        }

        // 出力ディレクトリの設定
        let outputDirectoryURL = try fileManager.createOutputDirectory(
            modelName: modelName,
            version: version,
            classificationMethod: classificationMethod,
            moduleOutputPath: outputParentDirPath
        )
        print("📁 出力ディレクトリ作成成功: \(outputDirectoryURL.path)")

        // トレーニングデータの準備
        let sourceDir = URL(fileURLWithPath: resourcesDirectoryPath)
        let positiveClass = classLabelDirURLs[1].lastPathComponent
        let negativeClass = classLabelDirURLs[0].lastPathComponent

        // 各クラスの画像ファイルを取得
        let positiveClassDir = sourceDir.appendingPathComponent(positiveClass)
        let negativeClassDir = sourceDir.appendingPathComponent(negativeClass)

        let positiveClassFiles = try FileManager.default.contentsOfDirectory(
            at: positiveClassDir,
            includingPropertiesForKeys: nil
        )

        let negativeClassFiles = try FileManager.default.contentsOfDirectory(
            at: negativeClassDir,
            includingPropertiesForKeys: nil
        )

        // クラスごとの画像数を更新
        classImageCounts[positiveClass] = positiveClassFiles.count
        classImageCounts[negativeClass] = negativeClassFiles.count

        print("📊 \(positiveClass): \(positiveClassFiles.count)枚, \(negativeClass): \(negativeClassFiles.count)枚")

        // トレーニングデータソースを作成
        let trainingDataSource = MLImageClassifier.DataSource.labeledDirectories(at: sourceDir)

        // モデルのトレーニング
        let trainingStartTime = Date()
        let imageClassifier = try MLImageClassifier(trainingData: trainingDataSource, parameters: modelParameters)
        let trainingEndTime = Date()
        let trainingDurationSeconds = trainingEndTime.timeIntervalSince(trainingStartTime)
        print("✅ モデルの作成が完了 (所要時間: \(String(format: "%.1f", trainingDurationSeconds))秒)")

        let currentTrainingMetrics = imageClassifier.trainingMetrics
        let currentValidationMetrics = imageClassifier.validationMetrics

        // 混同行列の計算
        let confusionMatrix = CICBinaryConfusionMatrix(
            dataTable: currentValidationMetrics.confusion,
            predictedColumn: "Predicted",
            actualColumn: "True Label",
            positiveClass: positiveClass
        )

        // 個別モデルのレポートを作成
        let modelFileName = "\(modelName)_\(classificationMethod)_\(version).mlmodel"
        let individualReport = CICIndividualModelReport(
            modelFileName: modelFileName,
            metrics: (
                training: (
                    accuracy: 1.0 - currentTrainingMetrics.classificationError,
                    errorRate: currentTrainingMetrics.classificationError
                ),
                validation: (
                    accuracy: 1.0 - currentValidationMetrics.classificationError,
                    errorRate: currentValidationMetrics.classificationError
                )
            ),
            confusionMatrix: confusionMatrix,
            classCounts: (
                positive: (name: positiveClass, count: positiveClassFiles.count),
                negative: (name: negativeClass, count: negativeClassFiles.count)
            )
        )

        // モデルのメタデータ作成
        let augmentationFinalDescription = if !modelParameters.augmentationOptions.isEmpty {
            String(describing: modelParameters.augmentationOptions)
        } else {
            "なし"
        }
        let featureExtractorDescription = String(describing: modelParameters.featureExtractor)

        let metricsDescription = createMetricsDescription(
            individualReport: individualReport,
            modelParameters: modelParameters,
            augmentationFinalDescription: augmentationFinalDescription,
            featureExtractorDescription: featureExtractorDescription
        )

        let modelMetadata = MLModelMetadata(
            author: author,
            shortDescription: metricsDescription,
            version: version
        )

        // モデルファイルを保存
        let modelFilePath = outputDirectoryURL.appendingPathComponent(individualReport.modelFileName).path
        print("💾 モデルファイル保存中: \(modelFilePath)")
        try imageClassifier.write(to: URL(fileURLWithPath: modelFilePath), metadata: modelMetadata)
        print("✅ モデルファイル保存完了")

        // メタデータの作成
        let metadata = CICTrainingMetadata(
            modelName: modelName,
            classLabelCounts: [
                individualReport.classCounts.negative.name: individualReport.classCounts.negative.count,
                individualReport.classCounts.positive.name: individualReport.classCounts.positive.count,
            ],
            maxIterations: modelParameters.maxIterations,
            dataAugmentationDescription: augmentationFinalDescription,
            featureExtractorDescription: featureExtractorDescription
        )

        let result = BinaryTrainingResult(
            metadata: metadata,
            metrics: (
                training: (
                    accuracy: 1.0 - imageClassifier.trainingMetrics.classificationError,
                    errorRate: imageClassifier.trainingMetrics.classificationError
                ),
                validation: (
                    accuracy: 1.0 - imageClassifier.validationMetrics.classificationError,
                    errorRate: imageClassifier.validationMetrics.classificationError
                )
            ),
            confusionMatrix: confusionMatrix,
            individualModelReport: individualReport
        )

        // 全モデルの比較表を表示
        result.displayComparisonTable()

        // ログを保存
        try result.saveLog(
            modelAuthor: author,
            modelName: modelName,
            modelVersion: version,
            outputDirPath: outputDirectoryURL.path
        )
    }

    private func createMetricsDescription(
        individualReport: CICIndividualModelReport,
        modelParameters: CreateML.MLImageClassifier.ModelParameters,
        augmentationFinalDescription: String,
        featureExtractorDescription: String
    ) -> String {
        var metricsDescription = """
        \(individualReport.classCounts.positive.name): \(individualReport.classCounts.positive.count)枚
        \(individualReport.classCounts.negative.name): \(individualReport.classCounts.negative.count)枚
        最大反復回数: \(modelParameters.maxIterations)回
        訓練正解率: \(String(format: "%.1f%%", individualReport.metrics.training.accuracy * 100.0))
        検証正解率: \(String(format: "%.1f%%", individualReport.metrics.validation.accuracy * 100.0))
        """

        if let confusionMatrix = individualReport.confusionMatrix {
            var metricsText = ""

            if confusionMatrix.recall.isFinite {
                metricsText += "再現率: \(String(format: "%.1f%%", confusionMatrix.recall * 100.0))\n"
            }
            if confusionMatrix.precision.isFinite {
                metricsText += "適合率: \(String(format: "%.1f%%", confusionMatrix.precision * 100.0))\n"
            }
            if confusionMatrix.f1Score.isFinite {
                metricsText += "F1スコア: \(String(format: "%.3f", confusionMatrix.f1Score))"
            }

            if !metricsText.isEmpty {
                metricsDescription += "\n" + metricsText
            }
        }

        metricsDescription += """

        データ拡張: \(augmentationFinalDescription)
        特徴抽出器: \(featureExtractorDescription)
        """

        return metricsDescription
    }

    public func setupOutputDirectory(modelName: String, version: String) throws -> URL {
        let outputDirectoryURL = try fileManager.createOutputDirectory(
            modelName: modelName,
            version: version,
            classificationMethod: classificationMethod,
            moduleOutputPath: outputParentDirPath
        )
        print("📁 出力ディレクトリ作成成功: \(outputDirectoryURL.path)")
        return outputDirectoryURL
    }

    public func getClassLabelDirectories() throws -> [URL] {
        let classLabelDirURLs = try fileManager.getClassLabelDirectories(resourcesPath: resourcesDirectoryPath)
        print("📁 検出されたクラスラベルディレクトリ: \(classLabelDirURLs.map(\.lastPathComponent).joined(separator: ", "))")

        guard classLabelDirURLs.count == 2 else {
            throw NSError(domain: "BinaryClassifier", code: -1, userInfo: [
                NSLocalizedDescriptionKey: "Binary分類には2つのクラスラベルディレクトリが必要です。現在 \(classLabelDirURLs.count)個。",
            ])
        }

        return classLabelDirURLs
    }
}

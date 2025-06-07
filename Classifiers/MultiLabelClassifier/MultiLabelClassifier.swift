import CICConfusionMatrix
import CICFileManager
import CICInterface
import CICTrainingResult
import Combine
import CoreML
import CreateML
import Foundation
import TabularData

public final class MultiLabelClassifier: ClassifierProtocol {
    public typealias TrainingResultType = MultiLabelTrainingResult

    private let fileManager = CICFileManager()
    public var outputDirectoryPathOverride: String?
    public var resourceDirPathOverride: String?
    private var classImageCounts: [String: Int] = [:]

    public var outputParentDirPath: String {
        if let override = outputDirectoryPathOverride {
            return override
        }
        let currentFileURL = URL(fileURLWithPath: #filePath)
        return currentFileURL
            .deletingLastPathComponent() // MultiLabelClassifier
            .deletingLastPathComponent() // Classifiers
            .deletingLastPathComponent() // Project root
            .appendingPathComponent("CICOutputModels")
            .appendingPathComponent("MultiLabelClassifier")
            .path
    }

    public var classificationMethod: String { "MultiLabel" }

    public var resourcesDirectoryPath: String {
        if let override = resourceDirPathOverride {
            return override
        }
        let currentFileURL = URL(fileURLWithPath: #filePath)
        return currentFileURL
            .deletingLastPathComponent() // MultiLabelClassifier
            .appendingPathComponent("Resources")
            .path
    }

    public init(
        outputDirectoryPathOverride: String? = nil,
        resourceDirPathOverride: String? = nil
    ) {
        self.outputDirectoryPathOverride = outputDirectoryPathOverride
        self.resourceDirPathOverride = resourceDirPathOverride
    }

    public func createAndSaveModel(
        author: String,
        modelName: String,
        version: String,
        modelParameters: CreateML.MLImageClassifier.ModelParameters
    ) throws {
        print("📁 リソースディレクトリ: \(resourcesDirectoryPath)")
        print("🚀 MultiLabelモデル作成開始 (バージョン: \(version))...")

        // クラスラベルディレクトリの取得
        let classLabelDirURLs = try getClassLabelDirectories()

        // トレーニングデータの準備
        print("📁 トレーニングデータ親ディレクトリ: \(resourcesDirectoryPath)")

        // 各クラスの画像ファイルを取得
        var allFiles: [URL] = []
        for classDir in classLabelDirURLs {
            let className = classDir.lastPathComponent
            let files = try FileManager.default.contentsOfDirectory(at: classDir, includingPropertiesForKeys: nil)
            classImageCounts[className] = files.count
            allFiles.append(contentsOf: files)
        }

        print("📊 合計画像枚数: \(allFiles.count)枚")
        for (className, count) in classImageCounts {
            print("📊 \(className): \(count)枚")
        }

        let trainingDataSource = MLImageClassifier.DataSource
            .labeledDirectories(at: URL(fileURLWithPath: resourcesDirectoryPath))
        print("📊 トレーニングデータソース作成完了")

        // モデルのトレーニング
        let trainingStartTime = Date()
        let imageClassifier = try MLImageClassifier(trainingData: trainingDataSource, parameters: modelParameters)
        let trainingEndTime = Date()
        let trainingDurationSeconds = trainingEndTime.timeIntervalSince(trainingStartTime)
        print("✅ モデルの作成が完了 (所要時間: \(String(format: "%.1f", trainingDurationSeconds))秒)")

        let trainingMetrics = imageClassifier.trainingMetrics
        let validationMetrics = imageClassifier.validationMetrics

        // 混同行列の計算
        let confusionMatrix = CICMultiClassConfusionMatrix(
            dataTable: validationMetrics.confusion,
            predictedColumn: "Predicted",
            actualColumn: "True Label"
        )

        // トレーニング結果の表示
        print("\n📊 トレーニング結果サマリー")
        print(String(
            format: "  訓練正解率: %.1f%%",
            (1.0 - trainingMetrics.classificationError) * 100.0
        ))

        if let confusionMatrix {
            print(String(
                format: "  検証正解率: %.1f%%",
                (1.0 - validationMetrics.classificationError) * 100.0
            ))
            print(confusionMatrix.getMatrixGraph())
        } else {
            print("⚠️ 警告: 検証データが不十分なため、混同行列の計算をスキップしました")
        }

        // モデルのメタデータ作成
        let augmentationFinalDescription = if !modelParameters.augmentationOptions.isEmpty {
            String(describing: modelParameters.augmentationOptions)
        } else {
            "なし"
        }
        let featureExtractorDescription = modelParameters.algorithm.description

        let metricsDescription = createMetricsDescription(
            individualReport: CICMultiClassModelReport(
                modelFileName: "\(modelName)_\(classificationMethod)_\(version).mlmodel",
                metrics: (
                    training: (
                        accuracy: 1.0 - trainingMetrics.classificationError,
                        errorRate: trainingMetrics.classificationError
                    ),
                    validation: (
                        accuracy: 1.0 - validationMetrics.classificationError,
                        errorRate: validationMetrics.classificationError
                    )
                ),
                confusionMatrix: confusionMatrix,
                classCounts: classImageCounts
            ),
            modelParameters: modelParameters,
            augmentationFinalDescription: augmentationFinalDescription,
            featureExtractorDescription: featureExtractorDescription
        )

        let modelMetadata = MLModelMetadata(
            author: author,
            shortDescription: metricsDescription,
            version: version
        )

        // 出力ディレクトリの設定
        let outputDirectoryURL = try fileManager.createOutputDirectory(
            modelName: modelName,
            version: version,
            classificationMethod: classificationMethod,
            moduleOutputPath: outputParentDirPath
        )
        print("📁 出力ディレクトリ作成成功: \(outputDirectoryURL.path)")

        // モデルファイルを保存
        let modelFileName = "\(modelName)_\(classificationMethod)_\(version).mlmodel"
        let modelFilePath = outputDirectoryURL.appendingPathComponent(modelFileName).path
        print("💾 モデルファイル保存中: \(modelFilePath)")
        try imageClassifier.write(to: URL(fileURLWithPath: modelFilePath), metadata: modelMetadata)
        print("✅ モデルファイル保存完了")

        // メタデータの作成
        let metadata = CICTrainingMetadata(
            modelName: modelName,
            classLabelCounts: classImageCounts,
            maxIterations: modelParameters.maxIterations,
            dataAugmentationDescription: augmentationFinalDescription,
            featureExtractorDescription: featureExtractorDescription
        )

        let result = MultiLabelTrainingResult(
            metadata: metadata,
            metrics: (
                training: (
                    accuracy: 1.0 - trainingMetrics.classificationError,
                    errorRate: trainingMetrics.classificationError
                ),
                validation: (
                    accuracy: 1.0 - validationMetrics.classificationError,
                    errorRate: validationMetrics.classificationError
                )
            ),
            confusionMatrix: confusionMatrix,
            individualModelReports: []
        )

        // 全モデルの比較表を表示
        result.displayComparisonTable()

        // ログを保存
        result.saveLog(
            modelAuthor: author,
            modelName: modelName,
            modelVersion: version,
            outputDirPath: outputDirectoryURL.path
        )
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

        guard classLabelDirURLs.count >= 2 else {
            throw NSError(domain: "MultiLabelClassifier", code: -1, userInfo: [
                NSLocalizedDescriptionKey: "MultiLabel分類には少なくとも2つのクラスラベルディレクトリが必要です。現在 \(classLabelDirURLs.count)個。",
            ])
        }

        return classLabelDirURLs
    }

    private func createMetricsDescription(
        individualReport: CICMultiClassModelReport,
        modelParameters _: CreateML.MLImageClassifier.ModelParameters,
        augmentationFinalDescription: String,
        featureExtractorDescription: String
    ) -> String {
        var metricsDescription = individualReport.generateMetricsDescription()

        metricsDescription += """

        データ拡張: \(augmentationFinalDescription)
        特徴抽出器: \(featureExtractorDescription)
        """

        return metricsDescription
    }
}

import CICConfusionMatrix
import CICFileManager
import CICInterface
import CICTrainingResult
import Combine
import CoreML
import CreateML
import Foundation
import TabularData

public final class OvOClassifier: ClassifierProtocol {
    public typealias TrainingResultType = OvOTrainingResult

    private let fileManager = CICFileManager()
    public var outputDirectoryPathOverride: String?
    public var resourceDirPathOverride: String?

    private static let imageExtensions = Set(["jpg", "jpeg", "png"])

    public var outputParentDirPath: String {
        if let override = outputDirectoryPathOverride {
            return override
        }
        let currentFileURL = URL(fileURLWithPath: #filePath)
        return currentFileURL
            .deletingLastPathComponent() // OvOClassifier
            .deletingLastPathComponent() // Classifiers
            .deletingLastPathComponent() // Project root
            .appendingPathComponent("CICOutputModels")
            .appendingPathComponent("OvOClassifier")
            .path
    }

    public var classificationMethod: String { "OvO" }

    public var resourcesDirectoryPath: String {
        if let override = resourceDirPathOverride {
            return override
        }
        let currentFileURL = URL(fileURLWithPath: #filePath)
        return currentFileURL
            .deletingLastPathComponent() // OvOClassifier
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

    static let tempBaseDirName = "TempOvOTrainingData"

    public func createAndSaveModel(
        author: String,
        modelName: String,
        version: String,
        modelParameters: CreateML.MLImageClassifier.ModelParameters,
        shouldEqualizeFileCount: Bool
    ) throws {
        print("📁 リソースディレクトリ: \(resourcesDirectoryPath)")
        print("🚀 OvOモデル作成開始 (バージョン: \(version))...")

        // 共通の説明文を作成
        let augmentationFinalDescription = if !modelParameters.augmentationOptions.isEmpty {
            String(describing: modelParameters.augmentationOptions)
        } else {
            "なし"
        }
        let featureExtractorDescription = modelParameters.algorithm.description

        // クラスラベルディレクトリの取得と検証
        let classLabelDirURLs = try fileManager.getClassLabelDirectories(resourcesPath: resourcesDirectoryPath)
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        print("📁 検出されたクラスラベルディレクトリ: \(classLabelDirURLs.map(\.lastPathComponent).joined(separator: ", "))")

        guard classLabelDirURLs.count >= 2 else {
            throw NSError(domain: "OvOClassifier", code: -1, userInfo: [
                NSLocalizedDescriptionKey: "OvO分類には少なくとも2つのクラスラベルディレクトリが必要です。現在 \(classLabelDirURLs.count)個。",
            ])
        }

        // クラスラベルの組み合わせを生成
        let classLabels = classLabelDirURLs.map(\.lastPathComponent)
        var combinations: [(String, String)] = []
        for i in 0 ..< classLabels.count {
            for j in (i + 1) ..< classLabels.count {
                combinations.append((classLabels[i], classLabels[j]))
            }
        }

        // 出力ディレクトリの設定
        let outputDirectoryURL = try fileManager.createOutputDirectory(
            modelName: modelName,
            version: version,
            classificationMethod: classificationMethod,
            moduleOutputPath: outputParentDirPath
        )
        print("📁 出力ディレクトリ作成成功: \(outputDirectoryURL.path)")

        // 各クラスの画像ファイル数を取得
        var classLabelCounts: [String: Int] = [:]
        for classLabel in classLabels {
            let classDir = URL(fileURLWithPath: resourcesDirectoryPath).appendingPathComponent(classLabel)
            let files = try fileManager.contentsOfDirectory(at: classDir, includingPropertiesForKeys: nil)
                .filter { Self.imageExtensions.contains($0.pathExtension.lowercased()) }
            classLabelCounts[classLabel] = files.count
        }

        // 各クラス組み合わせに対してモデルを生成
        var modelFilePaths: [String] = []
        var individualModelReports: [CICIndividualModelReport] = []

        for classPair in combinations {
            print("🔄 クラス組み合わせ [\(classPair.0) vs \(classPair.1)] のモデル作成開始...")

            let (imageClassifier, individualReport) = try createModelForClassPair(
                classPair: classPair,
                modelName: modelName,
                version: version,
                modelParameters: modelParameters,
                shouldEqualizeFileCount: shouldEqualizeFileCount
            )

            // モデルのメタデータ作成
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
            let modelFileName =
                "\(modelName)_\(classificationMethod)_\(classPair.0)_vs_\(classPair.1)_\(version).mlmodel"
            let modelFilePath = outputDirectoryURL.appendingPathComponent(modelFileName).path
            print("💾 モデルファイル保存中: \(modelFilePath)")
            try imageClassifier.write(to: URL(fileURLWithPath: modelFilePath), metadata: modelMetadata)
            print("✅ モデルファイル保存完了")

            individualModelReports.append(individualReport)
            modelFilePaths.append(modelFilePath)
        }

        // メタデータの作成
        let metadata = CICTrainingMetadata(
            modelName: modelName,
            classLabelCounts: classLabelCounts,
            maxIterations: modelParameters.maxIterations,
            dataAugmentationDescription: augmentationFinalDescription,
            featureExtractorDescription: featureExtractorDescription
        )

        let result = OvOTrainingResult(
            metadata: metadata,
            individualModelReports: individualModelReports
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

    private func createModelForClassPair(
        classPair: (String, String),
        modelName: String,
        version: String,
        modelParameters: CreateML.MLImageClassifier.ModelParameters,
        shouldEqualizeFileCount: Bool
    ) throws -> (MLImageClassifier, CICIndividualModelReport) {
        // トレーニングデータの準備
        let sourceDir = URL(fileURLWithPath: resourcesDirectoryPath)
        let class1Dir = sourceDir.appendingPathComponent(classPair.0)
        let class2Dir = sourceDir.appendingPathComponent(classPair.1)

        // バランス調整された画像セットを準備
        let balancedDirs = try fileManager.prepareEqualizedMinimumImageSet(
            classDirs: [class1Dir, class2Dir],
            shouldEqualize: shouldEqualizeFileCount
        )

        // トレーニングデータソースを作成
        guard let firstClassDir = balancedDirs[classPair.0] else {
            throw NSError(domain: "OvOClassifier", code: -1, userInfo: [
                NSLocalizedDescriptionKey: "トレーニングデータの準備に失敗しました。",
            ])
        }
        let trainingDataSource = MLImageClassifier.DataSource
            .labeledDirectories(at: firstClassDir.deletingLastPathComponent())

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
            positiveClass: classPair.1
        )

        // 個別モデルのレポートを作成
        let modelFileName = "\(modelName)_\(classificationMethod)_\(classPair.0)_vs_\(classPair.1)_\(version).mlmodel"
        let individualReport = try CICIndividualModelReport(
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
                positive: (
                    name: classPair.1,
                    count: fileManager.contentsOfDirectory(at: class2Dir, includingPropertiesForKeys: nil).count
                ),
                negative: (
                    name: classPair.0,
                    count: fileManager.contentsOfDirectory(at: class1Dir, includingPropertiesForKeys: nil).count
                )
            )
        )

        return (imageClassifier, individualReport)
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
}

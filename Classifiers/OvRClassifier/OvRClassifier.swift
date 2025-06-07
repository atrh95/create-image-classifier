import CICConfusionMatrix
import CICFileManager
import CICInterface
import CICTrainingResult
import Combine
import CoreML
import CreateML
import Foundation
import TabularData

public final class OvRClassifier: ClassifierProtocol {
    public typealias TrainingResultType = OvRTrainingResult

    private let fileManager = CICFileManager()
    public var outputDirectoryPathOverride: String?
    public var resourceDirPathOverride: String?

    private static let imageExtensions = Set(["jpg", "jpeg", "png"])
    public static let tempBaseDirName = "TempOvRTrainingData"

    public var outputParentDirPath: String {
        if let override = outputDirectoryPathOverride {
            return override
        }
        let currentFileURL = URL(fileURLWithPath: #filePath)
        return currentFileURL
            .deletingLastPathComponent() // OvRClassifier
            .deletingLastPathComponent() // Classifiers
            .deletingLastPathComponent() // Project root
            .appendingPathComponent("CICOutputModels")
            .appendingPathComponent("OvRClassifier")
            .path
    }

    public var classificationMethod: String { "OvR" }

    public var resourcesDirectoryPath: String {
        if let override = resourceDirPathOverride {
            return override
        }
        let currentFileURL = URL(fileURLWithPath: #filePath)
        return currentFileURL
            .deletingLastPathComponent() // OvRClassifier
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
        modelParameters: CreateML.MLImageClassifier.ModelParameters,
        shouldEqualizeFileCount _: Bool
    ) throws {
        print("📁 リソースディレクトリ: \(resourcesDirectoryPath)")
        print("🚀 OvRモデル作成開始 (バージョン: \(version))...")

        // クラスラベルディレクトリの取得
        let classLabelDirURLs = try fileManager.getClassLabelDirectories(resourcesPath: resourcesDirectoryPath)
        print("📁 検出されたクラスラベルディレクトリ: \(classLabelDirURLs.map(\.lastPathComponent).joined(separator: ", "))")

        guard classLabelDirURLs.count >= 2 else {
            throw NSError(domain: "OvRClassifier", code: -1, userInfo: [
                NSLocalizedDescriptionKey: "OvR分類には少なくとも2つのクラスラベルディレクトリが必要です。現在 \(classLabelDirURLs.count)個。",
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

        var individualModelReports: [CICIndividualModelReport] = []
        var classLabelCounts: [String: Int] = [:]

        // 各クラスに対して1つの .mlmodel を作成
        for oneClassDir in classLabelDirURLs {
            let oneClassLabel = oneClassDir.lastPathComponent
            print("🔄 クラス [\(oneClassLabel)] のモデル作成開始...")

            let (imageClassifier, individualReport) = try createModelForClass(
                oneClassLabel: oneClassLabel,
                modelName: modelName,
                version: version,
                modelParameters: modelParameters
            )

            // モデルのメタデータ作成
            let augmentationFinalDescription = if !modelParameters.augmentationOptions.isEmpty {
                String(describing: modelParameters.augmentationOptions)
            } else {
                "なし"
            }

            let featureExtractorDescription = modelParameters.algorithm.description

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

            individualModelReports.append(individualReport)
            classLabelCounts[oneClassLabel] = individualReport.classCounts.positive.count
        }

        let metadata = CICTrainingMetadata(
            modelName: modelName,
            classLabelCounts: classLabelCounts,
            maxIterations: modelParameters.maxIterations,
            dataAugmentationDescription: modelParameters.augmentationOptions
                .isEmpty ? "なし" : String(describing: modelParameters.augmentationOptions),
            featureExtractorDescription: modelParameters.algorithm.description
        )

        let result = OvRTrainingResult(
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

    private func createModelForClass(
        oneClassLabel: String,
        modelName: String,
        version: String,
        modelParameters: CreateML.MLImageClassifier.ModelParameters
    ) throws -> (MLImageClassifier, CICIndividualModelReport) {
        // トレーニングデータの準備
        let sourceDir = URL(fileURLWithPath: resourcesDirectoryPath)
        let positiveClassDir = sourceDir.appendingPathComponent(oneClassLabel)
        let trainingData = try prepareTrainingData(
            oneClassLabel: oneClassLabel,
            sourceDir: sourceDir,
            positiveClassDir: positiveClassDir
        )

        // トレーニングデータソースを作成
        let trainingDataSource = MLImageClassifier.DataSource.labeledDirectories(at: trainingData.tempDir)

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
            positiveClass: oneClassLabel
        )

        // 個別モデルのレポートを作成
        let modelFileName = "\(modelName)_\(classificationMethod)_\(oneClassLabel)_\(version).mlmodel"
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
                positive: (name: oneClassLabel, count: trainingData.positiveClassFiles.count),
                negative: (name: "rest", count: trainingData.restClassFiles.count)
            )
        )

        return (imageClassifier, individualReport)
    }

    private struct TrainingData {
        let positiveClassFiles: [URL]
        let restClassFiles: [URL]
        let tempDir: URL
    }

    private func prepareTrainingData(
        oneClassLabel: String,
        sourceDir: URL,
        positiveClassDir: URL
    ) throws -> TrainingData {
        let positiveClassFiles = try fileManager.contentsOfDirectory(
            at: positiveClassDir,
            includingPropertiesForKeys: nil
        )

        var restClassFiles: [URL] = []
        let subdirectories = try fileManager.contentsOfDirectory(
            at: sourceDir,
            includingPropertiesForKeys: [.isDirectoryKey]
        )
        .filter { $0.hasDirectoryPath && $0.lastPathComponent != oneClassLabel }

        let samplesPerRestClass = Int(ceil(Double(positiveClassFiles.count) / Double(subdirectories.count)))

        for subdir in subdirectories {
            let files = try fileManager.contentsOfDirectory(at: subdir, includingPropertiesForKeys: nil)
            let sampledFiles = files.shuffled().prefix(samplesPerRestClass)
            restClassFiles.append(contentsOf: sampledFiles)
        }
        print("📊 \(oneClassLabel) (正例): \(positiveClassFiles.count)枚, rest (計算値): \(restClassFiles.count)枚")

        let tempDir = fileManager.temporaryDirectory
            .appendingPathComponent(Self.tempBaseDirName)
            .appendingPathComponent(oneClassLabel) // クラス別のサブディレクトリを追加
        let tempPositiveDir = tempDir.appendingPathComponent(oneClassLabel)
        let tempRestDir = tempDir.appendingPathComponent("rest")

        if !fileManager.fileExists(atPath: tempDir.path) {
            try fileManager.createDirectory(at: tempDir, withIntermediateDirectories: true)
        }
        if fileManager.fileExists(atPath: tempPositiveDir.path) {
            try fileManager.removeItem(at: tempPositiveDir)
        }
        if fileManager.fileExists(atPath: tempRestDir.path) {
            try fileManager.removeItem(at: tempRestDir)
        }

        // Create fresh directories for copying
        try fileManager.createDirectory(at: tempPositiveDir, withIntermediateDirectories: true)
        try fileManager.createDirectory(at: tempRestDir, withIntermediateDirectories: true)

        // Copy positive class files
        for (index, file) in positiveClassFiles.enumerated() {
            let destination = tempPositiveDir.appendingPathComponent("\(index).\(file.pathExtension)")
            do {
                try fileManager.copyItem(at: file, to: destination)
            } catch {
                print(
                    "❌ 正例ファイルコピー失敗: \(file.lastPathComponent) -> \(destination.lastPathComponent) エラー: \(error.localizedDescription)"
                )
                // Consider re-throwing or failing early if a critical file fails to copy
            }
        }

        // Copy rest class files with unique names and robust error handling
        var copiedRestFilesCount = 0
        for file in restClassFiles {
            let originalDirectoryName = file.deletingLastPathComponent().lastPathComponent
            let originalFileName = file.lastPathComponent
            let uniqueDestinationFileName = "\(originalDirectoryName)_\(originalFileName)"
            let destination = tempRestDir.appendingPathComponent(uniqueDestinationFileName)

            do {
                try fileManager.copyItem(at: file, to: destination)
                copiedRestFilesCount += 1
            } catch {
                print(
                    "❌ restファイルコピー失敗: \(file.lastPathComponent) -> \(uniqueDestinationFileName) エラー: \(error.localizedDescription)"
                )
                // Log the error but continue if other files might succeed
            }
        }
        print("DEBUG: tempRestDirに実際にコピーされたrestファイル数: \(copiedRestFilesCount)枚")

        // Validate the actual count of files in tempRestDir immediately after copying
        let actualRestFilesCountInTempDir = try fileManager.contentsOfDirectory(
            at: tempRestDir,
            includingPropertiesForKeys: nil
        )
        .count
        print("DEBUG: tempRestDir内の実際に存在するrestファイル数 (確認): \(actualRestFilesCountInTempDir)枚")

        return TrainingData(
            positiveClassFiles: positiveClassFiles, // This refers to original files
            restClassFiles: restClassFiles, // This refers to original sampled files
            tempDir: tempDir
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
}

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
    private static let tempBaseDirName = "TempOvRTrainingData"

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
            .deletingLastPathComponent() // Classifiers
            .deletingLastPathComponent() // Project root
            .appendingPathComponent("CICResources")
            .appendingPathComponent("OvRResources")
            .path
    }

    public init(
        outputDirectoryPathOverride: String? = nil,
        resourceDirPathOverride: String? = nil
    ) {
        self.outputDirectoryPathOverride = outputDirectoryPathOverride
        self.resourceDirPathOverride = resourceDirPathOverride
    }

    public func create(
        author: String,
        modelName: String,
        version: String,
        modelParameters: CreateML.MLImageClassifier.ModelParameters
    ) async throws {
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
        for (index, oneClassDir) in classLabelDirURLs.enumerated() {
            let oneClassLabel = oneClassDir.lastPathComponent
            print("🔄 クラス [\(oneClassLabel)] のモデル作成開始...")

            let (imageClassifier, individualReport) = try await createModelForClass(
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

            individualModelReports.append(individualReport)
            classLabelCounts[oneClassLabel] = individualReport.classCounts.positive.count
        }

        let metadata = CICTrainingMetadata(
            modelName: modelName,
            classLabelCounts: classLabelCounts,
            maxIterations: modelParameters.maxIterations,
            dataAugmentationDescription: modelParameters.augmentationOptions
                .isEmpty ? "なし" : String(describing: modelParameters.augmentationOptions),
            featureExtractorDescription: String(describing: modelParameters.featureExtractor)
        )

        let result = OvRTrainingResult(
            metadata: metadata,
            individualModelReports: individualModelReports
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

    private func createModelForClass(
        oneClassLabel: String,
        modelName: String,
        version: String,
        modelParameters: CreateML.MLImageClassifier.ModelParameters
    ) async throws -> (MLImageClassifier, CICIndividualModelReport) {
        // トレーニングデータの準備
        let sourceDir = URL(fileURLWithPath: resourcesDirectoryPath)
        let positiveClassDir = sourceDir.appendingPathComponent(oneClassLabel)
        let (positiveClassFiles, restClassFiles, tempDir) = try prepareTrainingData(
            oneClassLabel: oneClassLabel,
            sourceDir: sourceDir,
            positiveClassDir: positiveClassDir
        )

        // トレーニングデータソースを作成
        let trainingDataSource = MLImageClassifier.DataSource.labeledDirectories(at: tempDir)

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
                positive: (name: oneClassLabel, count: positiveClassFiles.count),
                negative: (name: "rest", count: restClassFiles.count)
            )
        )

        return (imageClassifier, individualReport)
    }

    private func prepareTrainingData(
        oneClassLabel: String,
        sourceDir: URL,
        positiveClassDir: URL
    ) throws -> (positiveClassFiles: [URL], restClassFiles: [URL], tempDir: URL) {
        // 正例クラスの画像ファイルを取得
        let positiveClassFiles = try FileManager.default.contentsOfDirectory(
            at: positiveClassDir,
            includingPropertiesForKeys: nil
        )
        .filter { Self.imageExtensions.contains($0.pathExtension.lowercased()) }

        // 残りのクラスの画像URLを取得
        var restClassFiles: [URL] = []
        let subdirectories = try FileManager.default.contentsOfDirectory(
            at: sourceDir,
            includingPropertiesForKeys: [.isDirectoryKey]
        )
        .filter { $0.hasDirectoryPath && $0.lastPathComponent != oneClassLabel }

        // 各restクラスから均等にサンプリング
        let samplesPerRestClass = Int(ceil(Double(positiveClassFiles.count) / Double(subdirectories.count)))

        for subdir in subdirectories {
            let files = try FileManager.default.contentsOfDirectory(at: subdir, includingPropertiesForKeys: nil)
            let sampledFiles = files.shuffled().prefix(samplesPerRestClass)
            restClassFiles.append(contentsOf: sampledFiles)
        }
        print("📊 \(oneClassLabel): \(positiveClassFiles.count)枚, rest: \(restClassFiles.count)枚")

        // 一時ディレクトリを準備
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(Self.tempBaseDirName)
        let tempPositiveDir = tempDir.appendingPathComponent(oneClassLabel)
        let tempRestDir = tempDir.appendingPathComponent("rest")

        // 既存の一時ディレクトリをクリーンにする
        if FileManager.default.fileExists(atPath: tempDir.path) {
            try FileManager.default.removeItem(at: tempDir)
        }

        try FileManager.default.createDirectory(at: tempPositiveDir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: tempRestDir, withIntermediateDirectories: true)

        // 正例は全画像をコピー
        for (index, file) in positiveClassFiles.enumerated() {
            let destination = tempPositiveDir.appendingPathComponent("\(index).\(file.pathExtension)")
            try FileManager.default.copyItem(at: file, to: destination)
        }

        // 負例はサンプリング済みの画像をすべてコピー
        for (index, file) in restClassFiles.enumerated() {
            let destination = tempRestDir.appendingPathComponent("\(index).\(file.pathExtension)")
            try FileManager.default.copyItem(at: file, to: destination)
        }

        return (positiveClassFiles, restClassFiles, tempDir)
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

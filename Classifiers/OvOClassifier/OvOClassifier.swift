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

    public func create(
        author: String,
        modelName: String,
        version: String,
        modelParameters: CreateML.MLImageClassifier.ModelParameters
    ) async throws {
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
            let files = try FileManager.default.contentsOfDirectory(
                at: classDir,
                includingPropertiesForKeys: nil
            )
            .filter { Self.imageExtensions.contains($0.pathExtension.lowercased()) }
            classLabelCounts[classLabel] = files.count
        }

        // 各クラス組み合わせに対してモデルを生成
        var modelFilePaths: [String] = []
        var individualModelReports: [CICIndividualModelReport] = []

        for classPair in combinations {
            print("🔄 クラス組み合わせ [\(classPair.0) vs \(classPair.1)] のモデル作成開始...")

            let (imageClassifier, individualReport) = try await createModelForClassPair(
                classPair: classPair,
                modelName: modelName,
                version: version,
                modelParameters: modelParameters
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
        modelParameters: CreateML.MLImageClassifier.ModelParameters
    ) async throws -> (MLImageClassifier, CICIndividualModelReport) {
        // トレーニングデータの準備
        let sourceDir = URL(fileURLWithPath: resourcesDirectoryPath)
        let class1Dir = sourceDir.appendingPathComponent(classPair.0)
        let class2Dir = sourceDir.appendingPathComponent(classPair.1)
        let trainingData = try prepareTrainingData(
            classPair: classPair,
            sourceDir: sourceDir,
            class1Dir: class1Dir,
            class2Dir: class2Dir
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
            positiveClass: classPair.1
        )

        // 個別モデルのレポートを作成
        let modelFileName = "\(modelName)_\(classificationMethod)_\(classPair.0)_vs_\(classPair.1)_\(version).mlmodel"
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
                positive: (name: classPair.1, count: trainingData.class2Files.count),
                negative: (name: classPair.0, count: trainingData.class1Files.count)
            )
        )

        return (imageClassifier, individualReport)
    }

    private struct TrainingData {
        let class1Files: [URL]
        let class2Files: [URL]
        let tempDir: URL
    }

    private func prepareTrainingData(
        classPair: (String, String),
        sourceDir _: URL,
        class1Dir: URL,
        class2Dir: URL
    ) throws -> TrainingData {
        // 各クラスの画像ファイルを取得
        let class1Files = try FileManager.default.contentsOfDirectory(
            at: class1Dir,
            includingPropertiesForKeys: nil
        )
        .filter { Self.imageExtensions.contains($0.pathExtension.lowercased()) }

        let class2Files = try FileManager.default.contentsOfDirectory(
            at: class2Dir,
            includingPropertiesForKeys: nil
        )
        .filter { Self.imageExtensions.contains($0.pathExtension.lowercased()) }

        // 最小枚数を取得
        let minCount = min(class1Files.count, class2Files.count)

        // 各クラスから最小枚数分の画像をランダムに選択
        let selectedClass1Files = Array(class1Files.shuffled().prefix(minCount))
        let selectedClass2Files = Array(class2Files.shuffled().prefix(minCount))

        print("📊 \(classPair.0): \(selectedClass1Files.count)枚, \(classPair.1): \(selectedClass2Files.count)枚")

        // 一時ディレクトリを準備
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(Self.tempBaseDirName)
        let tempClass1Dir = tempDir.appendingPathComponent(classPair.0)
        let tempClass2Dir = tempDir.appendingPathComponent(classPair.1)

        // 既存の一時ディレクトリをクリーンにする
        if FileManager.default.fileExists(atPath: tempDir.path) {
            try FileManager.default.removeItem(at: tempDir)
        }

        try FileManager.default.createDirectory(at: tempClass1Dir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: tempClass2Dir, withIntermediateDirectories: true)

        // 各クラスの画像をコピー
        for (index, file) in selectedClass1Files.enumerated() {
            let destination = tempClass1Dir.appendingPathComponent("\(index).\(file.pathExtension)")
            try FileManager.default.copyItem(at: file, to: destination)
        }

        for (index, file) in selectedClass2Files.enumerated() {
            let destination = tempClass2Dir.appendingPathComponent("\(index).\(file.pathExtension)")
            try FileManager.default.copyItem(at: file, to: destination)
        }

        return TrainingData(
            class1Files: selectedClass1Files,
            class2Files: selectedClass2Files,
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

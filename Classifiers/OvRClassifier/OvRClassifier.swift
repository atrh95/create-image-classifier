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

    static let tempBaseDirName = "TempOvRTrainingData"

    public func create(
        author: String,
        modelName: String,
        version: String,
        modelParameters: CreateML.MLImageClassifier.ModelParameters
    ) async throws -> OvRTrainingResult {
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

        var individualModelReports: [CICIndividualModelReport] = []
        var firstModelTrainingMetrics: MLClassifierMetrics?
        var firstModelValidationMetrics: MLClassifierMetrics?
        var modelFilePaths: [String] = []

        // 出力ディレクトリの設定
        let outputDirectoryURL = try fileManager.createOutputDirectory(
            modelName: modelName,
            version: version,
            classificationMethod: classificationMethod,
            moduleOutputPath: outputParentDirPath
        )
        print("📁 出力ディレクトリ作成成功: \(outputDirectoryURL.path)")

        // 各クラスに対して1つのモデルを作成
        for (index, oneClassDir) in classLabelDirURLs.enumerated() {
            let oneClassLabel = oneClassDir.lastPathComponent
            print("🔄 クラス [\(oneClassLabel)] のモデル作成開始...")

            // トレーニングデータの準備
            let sourceDir = URL(fileURLWithPath: resourcesDirectoryPath)
            let positiveClassDir = sourceDir.appendingPathComponent(oneClassLabel)

            // 正例クラスの画像ファイルを取得
            let positiveClassFiles = try FileManager.default.contentsOfDirectory(
                at: positiveClassDir,
                includingPropertiesForKeys: nil
            )
            .filter { Self.imageExtensions.contains($0.pathExtension.lowercased()) }

            // 負例クラスの画像ファイルを取得
            var negativeClassFiles: [URL] = []
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
                negativeClassFiles.append(contentsOf: sampledFiles)
            }
            print("📊 \(oneClassLabel): \(positiveClassFiles.count)枚, rest: \(negativeClassFiles.count)枚")

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
            for (index, file) in negativeClassFiles.enumerated() {
                let destination = tempRestDir.appendingPathComponent("\(index).\(file.pathExtension)")
                try FileManager.default.copyItem(at: file, to: destination)
            }

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

            // 最初のモデルのメトリクスを保存
            if firstModelTrainingMetrics == nil {
                firstModelTrainingMetrics = currentTrainingMetrics
                firstModelValidationMetrics = currentValidationMetrics
            }

            // モデルのメタデータ作成
            let augmentationFinalDescription = if !modelParameters.augmentationOptions.isEmpty {
                String(describing: modelParameters.augmentationOptions)
            } else {
                "なし"
            }

            let featureExtractorDescription = String(describing: modelParameters.featureExtractor)

            var metricsDescription = """
            \(oneClassLabel): \(positiveClassFiles.count)枚
            Rest: \(negativeClassFiles.count)枚
            訓練正解率: \(String(format: "%.1f%%", (1.0 - currentTrainingMetrics.classificationError) * 100.0))
            検証正解率: \(String(format: "%.1f%%", (1.0 - currentValidationMetrics.classificationError) * 100.0))
            """

            // 現在のクラスのメトリクスのみを計算
            let confusionMatrix = CICBinaryConfusionMatrix(
                dataTable: currentValidationMetrics.confusion,
                predictedColumn: "Predicted",
                actualColumn: "True Label",
                positiveClass: oneClassLabel
            )

            if let confusionMatrix {
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

            let modelMetadata = MLModelMetadata(
                author: author,
                shortDescription: metricsDescription,
                version: version
            )

            // モデルファイル名を生成
            let modelFileName = "\(modelName)_\(classificationMethod)_\(oneClassLabel)_\(version).mlmodel"

            // モデルファイルを保存
            let modelFilePath = outputDirectoryURL.appendingPathComponent(modelFileName).path
            print("💾 モデルファイル保存中: \(modelFilePath)")
            try imageClassifier.write(to: URL(fileURLWithPath: modelFilePath), metadata: modelMetadata)
            print("✅ モデルファイル保存完了")
            modelFilePaths.append(modelFilePath)

            // 個別モデルのレポートを作成
            let individualReport = CICIndividualModelReport(
                modelName: modelFileName,
                positiveClassName: oneClassLabel,
                negativeClassName: "rest",
                trainingAccuracyRate: 1.0 - currentTrainingMetrics.classificationError,
                validationAccuracyRate: 1.0 - currentValidationMetrics.classificationError,
                confusionMatrix: confusionMatrix
            )
            individualModelReports.append(individualReport)
        }

        // 最初のモデルのメトリクスを使用してトレーニング結果を作成
        guard let firstModelTrainingMetrics,
              let firstModelValidationMetrics
        else {
            throw NSError(
                domain: "OvRClassifier",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Training failed"]
            )
        }

        // 全モデルの比較表を表示
        print("\n📊 全モデルの性能")
        print(
            "+------------------+------------------+------------------+------------------+------------------+------------------+"
        )
        print("| クラス           | 訓練正解率       | 検証正解率       | 再現率           | 適合率           | F1スコア         |")
        print(
            "+------------------+------------------+------------------+------------------+------------------+------------------+"
        )
        for report in individualModelReports {
            let recall = report.confusionMatrix?.recall ?? 0.0
            let precision = report.confusionMatrix?.precision ?? 0.0
            let f1Score = report.confusionMatrix?.f1Score ?? 0.0
            print(
                "| \(report.positiveClassName.padding(toLength: 16, withPad: " ", startingAt: 0)) | \(String(format: "%14.1f%%", report.trainingAccuracyRate * 100.0)) | \(String(format: "%14.1f%%", report.validationAccuracyRate * 100.0)) | \(String(format: "%14.1f%%", recall * 100.0)) | \(String(format: "%14.1f%%", precision * 100.0)) | \(String(format: "%14.3f", f1Score)) |"
            )
        }
        print(
            "+------------------+------------------+------------------+------------------+------------------+------------------+"
        )

        let augmentationFinalDescription = if !modelParameters.augmentationOptions.isEmpty {
            String(describing: modelParameters.augmentationOptions)
        } else {
            "なし"
        }

        let featureExtractorDescription = String(describing: modelParameters.featureExtractor)

        let metadata = CICTrainingMetadata(
            modelName: modelName,
            trainingDurationInSeconds: 0,
            trainedModelFilePath: modelFilePaths[0],
            detectedClassLabelsList: classLabelDirURLs.map(\.lastPathComponent),
            maxIterations: modelParameters.maxIterations,
            dataAugmentationDescription: augmentationFinalDescription,
            featureExtractorDescription: featureExtractorDescription
        )

        return OvRTrainingResult(
            metadata: metadata,
            individualModelReports: individualModelReports
        )
    }
}

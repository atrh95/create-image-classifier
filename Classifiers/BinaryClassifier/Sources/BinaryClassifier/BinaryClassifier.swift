import CICConfusionMatrix
import CICFileManager
import CICInterface
import CICTrainingResult
import CoreML
import CreateML
import Foundation

public final class BinaryClassifier: ClassifierProtocol {
    public typealias TrainingResultType = BinaryTrainingResult

    private let fileManager = CICFileManager()
    public var outputDirectoryPathOverride: String?
    public var resourceDirPathOverride: String?
    private var classImageCounts: [String: Int] = [:]

    private static let tempBaseDirName = "TempBinaryTrainingData"

    public var outputParentDirPath: String {
        if let override = outputDirectoryPathOverride {
            return override
        }
        let currentFileURL = URL(fileURLWithPath: #filePath)
        return currentFileURL
            .deletingLastPathComponent() // BinaryClassifier
            .deletingLastPathComponent() // Sources
            .deletingLastPathComponent() // BinaryClassifier
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
            .deletingLastPathComponent() // BinaryClassifier
            .appendingPathComponent("Resources")
            .path
    }

    public var classificationMethod: String { "Binary" }

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
        modelParameters: MLImageClassifier.ModelParameters,
        shouldEqualizeFileCount: Bool
    ) throws {
        print("📁 リソースディレクトリ: \(resourcesDirectoryPath)")
        print("🚀 Binaryモデル作成開始 (バージョン: \(version))...")

        // 共通の説明文を作成
        let augmentationFinalDescription = if !modelParameters.augmentationOptions.isEmpty {
            String(describing: modelParameters.augmentationOptions)
        } else {
            "なし"
        }
        let featureExtractorDescription = modelParameters.algorithm.description

        // クラスラベルディレクトリの取得と検証
        let classLabelDirURLs = try getClassLabelDirectories()
        print("📁 検出されたクラスラベルディレクトリ: \(classLabelDirURLs.map(\.lastPathComponent).joined(separator: ", "))")

        guard classLabelDirURLs.count == 2 else {
            throw NSError(domain: "BinaryClassifier", code: -1, userInfo: [
                NSLocalizedDescriptionKey: "Binary分類には2つのクラスラベルディレクトリが必要です。現在 \(classLabelDirURLs.count)個。",
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

        // 各クラスの画像ファイル数を取得
        var classLabelCounts: [String: Int] = [:]
        for classDirURL in classLabelDirURLs {
            let files = try FileManager.default.contentsOfDirectory(
                at: classDirURL,
                includingPropertiesForKeys: nil
            )
            classLabelCounts[classDirURL.lastPathComponent] = files.count
        }

        // バランス調整された画像セットを準備
        let balancedDirs = try fileManager.prepareEqualizedMinimumImageSet(
            classDirs: classLabelDirURLs,
            shouldEqualize: shouldEqualizeFileCount
        )

        // トレーニングデータソースを作成
        guard let firstClassDir = balancedDirs[classLabelDirURLs[0].lastPathComponent] else {
            throw NSError(domain: "BinaryClassifier", code: -1, userInfo: [
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
            positiveClass: classLabelDirURLs[1].lastPathComponent
        )

        // 個別モデルのレポートを作成
        let modelFileName =
            "\(modelName)_\(classificationMethod)_\(classLabelDirURLs[0].lastPathComponent)_vs_\(classLabelDirURLs[1].lastPathComponent)_\(version).mlmodel"
        guard let positiveClassDir = balancedDirs[classLabelDirURLs[1].lastPathComponent],
              let negativeClassDir = balancedDirs[classLabelDirURLs[0].lastPathComponent]
        else {
            throw NSError(domain: "BinaryClassifier", code: -1, userInfo: [
                NSLocalizedDescriptionKey: "トレーニングデータの準備に失敗しました。",
            ])
        }
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
                    name: classLabelDirURLs[1].lastPathComponent,
                    count: FileManager.default
                        .contentsOfDirectory(at: positiveClassDir, includingPropertiesForKeys: nil).count
                ),
                negative: (
                    name: classLabelDirURLs[0].lastPathComponent,
                    count: FileManager.default
                        .contentsOfDirectory(at: negativeClassDir, includingPropertiesForKeys: nil).count
                )
            )
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
        let modelFilePath = outputDirectoryURL.appendingPathComponent(modelFileName).path
        print("💾 モデルファイル保存中: \(modelFilePath)")
        try imageClassifier.write(to: URL(fileURLWithPath: modelFilePath), metadata: modelMetadata)
        print("✅ モデルファイル保存完了")

        // メタデータの作成
        let metadata = CICTrainingMetadata(
            modelName: modelName,
            classLabelCounts: classLabelCounts,
            maxIterations: modelParameters.maxIterations,
            dataAugmentationDescription: augmentationFinalDescription,
            featureExtractorDescription: featureExtractorDescription
        )

        let result = BinaryTrainingResult(
            metadata: metadata,
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
            individualModelReport: individualReport
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

    private func createMetricsDescription(
        individualReport: CICIndividualModelReport,
        modelParameters: CreateML.MLImageClassifier.ModelParameters,
        augmentationFinalDescription: String,
        featureExtractorDescription: String
    ) -> String {
        var metricsDescription = """
        Positive: \(individualReport.classCounts.positive.name) (\(individualReport.classCounts.positive.count)枚)
        Negative: \(individualReport.classCounts.negative.name) (\(individualReport.classCounts.negative.count)枚)
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
        // positive/negative構造をチェック
        let resourcesURL = URL(fileURLWithPath: resourcesDirectoryPath)
        let positiveURL = resourcesURL.appendingPathComponent("positive")
        let negativeURL = resourcesURL.appendingPathComponent("negative")

        if FileManager.default.fileExists(atPath: positiveURL.path),
           FileManager.default.fileExists(atPath: negativeURL.path) {
            // positive/negative構造の場合
            return try getClassLabelDirectoriesFromPositiveNegativeStructure()
        } else {
            // 従来の構造の場合
            return try getClassLabelDirectoriesFromLegacyStructure()
        }
    }

    private func getClassLabelDirectoriesFromPositiveNegativeStructure() throws -> [URL] {
        let resourcesURL = URL(fileURLWithPath: resourcesDirectoryPath)
        let positiveURL = resourcesURL.appendingPathComponent("positive")
        let negativeURL = resourcesURL.appendingPathComponent("negative")

        // positiveディレクトリ内のサブディレクトリを取得
        let positiveContents = try FileManager.default.contentsOfDirectory(
            at: positiveURL,
            includingPropertiesForKeys: [.isDirectoryKey]
        ).filter { url in
            var isDirectory: ObjCBool = false
            FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory)
            return isDirectory.boolValue
        }

        // negativeディレクトリ内のサブディレクトリを取得
        let negativeContents = try FileManager.default.contentsOfDirectory(
            at: negativeURL,
            includingPropertiesForKeys: [.isDirectoryKey]
        ).filter { url in
            var isDirectory: ObjCBool = false
            FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory)
            return isDirectory.boolValue
        }

        // 各ディレクトリに一つずつのサブディレクトリがあることを確認
        guard positiveContents.count == 1 else {
            throw NSError(domain: "BinaryClassifier", code: -1, userInfo: [
                NSLocalizedDescriptionKey: "positiveディレクトリには1つのクラスディレクトリが必要です。現在 \(positiveContents.count)個。",
            ])
        }

        guard negativeContents.count == 1 else {
            throw NSError(domain: "BinaryClassifier", code: -1, userInfo: [
                NSLocalizedDescriptionKey: "negativeディレクトリには1つのクラスディレクトリが必要です。現在 \(negativeContents.count)個。",
            ])
        }

        let positiveClassURL = positiveContents[0]
        let negativeClassURL = negativeContents[0]

        print("📁 Positive/Negative構造を検出:")
        print("  ✅ Positive: \(positiveClassURL.lastPathComponent)")
        print("  🔴 Negative: \(negativeClassURL.lastPathComponent)")

        // positiveクラスを最初に、negativeクラスを2番目に返す（order matters for confusion matrix）
        return [negativeClassURL, positiveClassURL]
    }

    private func getClassLabelDirectoriesFromLegacyStructure() throws -> [URL] {
        let classLabelDirURLs = try fileManager.getClassLabelDirectories(resourcesPath: resourcesDirectoryPath)
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        print("📁 従来構造を検出 - クラスラベルディレクトリ: \(classLabelDirURLs.map(\.lastPathComponent).joined(separator: ", "))")
        print("⚠️  警告: アルファベット順で2番目のディレクトリ (\(classLabelDirURLs[1].lastPathComponent)) がpositiveクラスとして扱われます")

        guard classLabelDirURLs.count == 2 else {
            throw NSError(domain: "BinaryClassifier", code: -1, userInfo: [
                NSLocalizedDescriptionKey: "Binary分類には2つのクラスラベルディレクトリが必要です。現在 \(classLabelDirURLs.count)個。",
            ])
        }

        return classLabelDirURLs
    }
}

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
            .deletingLastPathComponent() // Classifiers
            .deletingLastPathComponent() // Project root
            .appendingPathComponent("CICResources")
            .appendingPathComponent("MultiLabelResources")
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
    ) async -> MultiLabelTrainingResult? {
        print("📁 リソースディレクトリ: \(resourcesDirectoryPath)")
        print("🚀 MultiLabelモデル作成開始 (バージョン: \(version))...")

        do {
            // クラスラベルディレクトリの取得
            let classLabelDirURLs = try getClassLabelDirectories()

            // トレーニングデータの準備
            let trainingDataSource = try prepareTrainingData(from: classLabelDirURLs)
            print("📊 トレーニングデータソース作成完了")

            // モデルのトレーニング
            let (imageClassifier, trainingDurationSeconds) = try trainModel(
                trainingDataSource: trainingDataSource,
                modelParameters: modelParameters
            )

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
            let modelMetadata = createModelMetadata(
                author: author,
                version: version,
                classLabelDirURLs: classLabelDirURLs,
                trainingMetrics: trainingMetrics,
                validationMetrics: validationMetrics,
                modelParameters: modelParameters
            )

            // 出力ディレクトリの設定
            let outputDirectoryURL = try setupOutputDirectory(modelName: modelName, version: version)

            let modelFilePath = try saveMLModel(
                imageClassifier: imageClassifier,
                modelName: modelName,
                modelFileName: "\(modelName)_\(classificationMethod)_\(version).mlmodel",
                version: version,
                outputDirectoryURL: outputDirectoryURL,
                metadata: modelMetadata
            )

            // モデルの性能を表示
            print("\n📊 モデルの性能")
            print("+------------------+------------------+------------------+------------------+------------------+")
            print("| 訓練正解率       | 検証正解率       | 再現率           | 適合率           | F1スコア         |")
            print("+------------------+------------------+------------------+------------------+------------------+")

            let classMetrics = confusionMatrix?.calculateMetrics() ?? []
            let macroRecall = classMetrics.map(\.recall).reduce(0.0, +) / Double(max(1, classMetrics.count))
            let macroPrecision = classMetrics.map(\.precision).reduce(0.0, +) / Double(max(1, classMetrics.count))
            let macroF1Score = classMetrics.map(\.f1Score).reduce(0.0, +) / Double(max(1, classMetrics.count))

            print(
                "| \(String(format: "%14.1f%%", (1.0 - trainingMetrics.classificationError) * 100.0)) | \(String(format: "%14.1f%%", (1.0 - validationMetrics.classificationError) * 100.0)) | \(String(format: "%14.1f%%", macroRecall * 100.0)) | \(String(format: "%14.1f%%", macroPrecision * 100.0)) | \(String(format: "%14.3f", macroF1Score)) |"
            )
            print("+------------------+------------------+------------------+------------------+------------------+")

            return createTrainingResult(
                modelName: modelName,
                classLabelDirURLs: classLabelDirURLs,
                trainingMetrics: trainingMetrics,
                validationMetrics: validationMetrics,
                modelParameters: modelParameters,
                trainingDurationSeconds: trainingDurationSeconds,
                modelFilePath: modelFilePath
            )

        } catch let createMLError as CreateML.MLCreateError {
            print("🛑 エラー: モデル [\(modelName)] のトレーニングまたは保存失敗 (CreateML): \(createMLError.localizedDescription)")
            print("詳細なエラー情報:")
            print("- エラーコード: \(createMLError.errorCode)")
            print("- エラーの種類: \(type(of: createMLError))")
            return nil
        } catch {
            print("🛑 エラー: トレーニングプロセス中に予期しないエラー: \(error.localizedDescription)")
            print("エラーの詳細:")
            print(error)
            return nil
        }
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

    public func prepareTrainingData(from classLabelDirURLs: [URL]) throws -> MLImageClassifier.DataSource {
        print("📁 トレーニングデータ親ディレクトリ: \(resourcesDirectoryPath)")

        let imageExtensions = Set(["jpg", "jpeg", "png"])

        // 各クラスの画像ファイルを取得
        var allFiles: [URL] = []
        for classDir in classLabelDirURLs {
            let files = try FileManager.default.contentsOfDirectory(at: classDir, includingPropertiesForKeys: nil)
                .filter { imageExtensions.contains($0.pathExtension.lowercased()) }
            allFiles.append(contentsOf: files)
        }

        print("📊 合計画像枚数: \(allFiles.count)枚")

        return MLImageClassifier.DataSource.labeledDirectories(at: URL(fileURLWithPath: resourcesDirectoryPath))
    }

    public func trainModel(
        trainingDataSource: MLImageClassifier.DataSource,
        modelParameters: CreateML.MLImageClassifier.ModelParameters
    ) throws -> (MLImageClassifier, TimeInterval) {
        print("🔄 モデルトレーニング開始...")
        let trainingStartTime = Date()
        let imageClassifier = try MLImageClassifier(trainingData: trainingDataSource, parameters: modelParameters)
        let trainingEndTime = Date()
        let trainingDurationSeconds = trainingEndTime.timeIntervalSince(trainingStartTime)
        print("✅ モデルトレーニング完了 (所要時間: \(String(format: "%.1f", trainingDurationSeconds))秒)")
        return (imageClassifier, trainingDurationSeconds)
    }

    public func createModelMetadata(
        author: String,
        version: String,
        classLabelDirURLs: [URL],
        trainingMetrics: MLClassifierMetrics,
        validationMetrics: MLClassifierMetrics,
        modelParameters: CreateML.MLImageClassifier.ModelParameters
    ) -> MLModelMetadata {
        let augmentationFinalDescription = if !modelParameters.augmentationOptions.isEmpty {
            String(describing: modelParameters.augmentationOptions)
        } else {
            "なし"
        }

        let featureExtractorDescription = String(describing: modelParameters.featureExtractor)

        // 混同行列から再現率と適合率を計算
        let confusionMatrix = CICMultiClassConfusionMatrix(
            dataTable: validationMetrics.confusion,
            predictedColumn: "Predicted",
            actualColumn: "True Label"
        )

        var metricsDescription = """
        クラス: \(classLabelDirURLs.map(\.lastPathComponent).joined(separator: ", "))
        訓練正解率: \(String(format: "%.1f%%", (1.0 - trainingMetrics.classificationError) * 100.0))
        検証正解率: \(String(format: "%.1f%%", (1.0 - validationMetrics.classificationError) * 100.0))
        """

        if let confusionMatrix {
            let classMetrics = confusionMatrix.calculateMetrics()
            let macroRecall = classMetrics.map(\.recall).reduce(0.0, +) / Double(classMetrics.count)
            let macroPrecision = classMetrics.map(\.precision).reduce(0.0, +) / Double(classMetrics.count)
            let macroF1Score = classMetrics.map(\.f1Score).reduce(0.0, +) / Double(classMetrics.count)
            metricsDescription += """

            マクロ平均再現率: \(String(format: "%.1f%%", macroRecall * 100.0))
            マクロ平均適合率: \(String(format: "%.1f%%", macroPrecision * 100.0))
            マクロ平均F1スコア: \(String(format: "%.3f", macroF1Score))
            """
        }

        metricsDescription += """

        データ拡張: \(augmentationFinalDescription)
        特徴抽出器: \(featureExtractorDescription)
        """

        return MLModelMetadata(
            author: author,
            shortDescription: metricsDescription,
            version: version
        )
    }

    public func saveMLModel(
        imageClassifier: MLImageClassifier,
        modelName _: String,
        modelFileName: String,
        version _: String,
        outputDirectoryURL: URL,
        metadata: MLModelMetadata
    ) throws -> String {
        let modelFilePath = outputDirectoryURL.appendingPathComponent(modelFileName).path

        print("💾 モデルファイル保存中: \(modelFilePath)")
        try imageClassifier.write(to: URL(fileURLWithPath: modelFilePath), metadata: metadata)
        print("✅ モデルファイル保存完了")

        return modelFilePath
    }

    public func createTrainingResult(
        modelName: String,
        classLabelDirURLs: [URL],
        trainingMetrics: MLClassifierMetrics,
        validationMetrics: MLClassifierMetrics,
        modelParameters: CreateML.MLImageClassifier.ModelParameters,
        trainingDurationSeconds: TimeInterval,
        modelFilePath: String
    ) -> MultiLabelTrainingResult {
        let augmentationFinalDescription = if !modelParameters.augmentationOptions.isEmpty {
            String(describing: modelParameters.augmentationOptions)
        } else {
            "なし"
        }

        let featureExtractorDescription = String(describing: modelParameters.featureExtractor)

        let metadata = CICTrainingMetadata(
            modelName: modelName,
            trainingDurationInSeconds: trainingDurationSeconds,
            trainedModelFilePath: modelFilePath,
            detectedClassLabelsList: classLabelDirURLs.map(\.lastPathComponent),
            maxIterations: modelParameters.maxIterations,
            dataAugmentationDescription: augmentationFinalDescription,
            featureExtractorDescription: featureExtractorDescription
        )

        let confusionMatrix = CICMultiClassConfusionMatrix(
            dataTable: validationMetrics.confusion,
            predictedColumn: "Predicted",
            actualColumn: "True Label"
        )

        return MultiLabelTrainingResult(
            metadata: metadata,
            trainingMetrics: (
                accuracy: 1.0 - trainingMetrics.classificationError,
                errorRate: trainingMetrics.classificationError
            ),
            validationMetrics: (
                accuracy: 1.0 - validationMetrics.classificationError,
                errorRate: validationMetrics.classificationError
            ),
            confusionMatrix: confusionMatrix,
            individualModelReports: []
        )
    }
}

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
    public var testResourcesDirectoryPath: String?

    public var outputDirPath: String {
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
        if let testPath = testResourcesDirectoryPath {
            return testPath
        }
        let currentFileURL = URL(fileURLWithPath: #filePath)
        return currentFileURL
            .deletingLastPathComponent() // OvOClassifier
            .deletingLastPathComponent() // Classifiers
            .deletingLastPathComponent() // Project root
            .appendingPathComponent("CICResources")
            .appendingPathComponent("OvOResources")
            .path
    }

    public init(outputDirectoryPathOverride: String? = nil) {
        self.outputDirectoryPathOverride = outputDirectoryPathOverride
    }

    static let tempBaseDirName = "TempOvOTrainingData"

    public func train(
        author: String,
        modelName: String,
        version: String,
        modelParameters: CreateML.MLImageClassifier.ModelParameters,
        scenePrintRevision: Int?
    ) async -> OvOTrainingResult? {
        print("📁 リソースディレクトリ: \(resourcesDirectoryPath)")
        print("🚀 OvOトレーニング開始 (バージョン: \(version))...")

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
                modelParameters: modelParameters,
                scenePrintRevision: scenePrintRevision
            )

            // 出力ディレクトリの設定
            let outputDirectoryURL = try setupOutputDirectory(modelName: modelName, version: version)

            // クラスラベルを取得してファイル名を生成
            let classLabels = classLabelDirURLs.map { $0.lastPathComponent }

            // OvOの場合は、クラス間の対戦を表す形式に変換
            let classLabelsString = classLabels.joined(separator: "_vs_")

            let modelFileName = "\(modelName)_\(classificationMethod)_\(classLabelsString)_\(version).mlmodel"

            let modelFilePath = try saveModel(
                imageClassifier: imageClassifier,
                modelName: modelName,
                modelFileName: modelFileName,
                version: version,
                outputDirectoryURL: outputDirectoryURL,
                metadata: modelMetadata
            )

            return createTrainingResult(
                modelName: modelName,
                classLabelDirURLs: classLabelDirURLs,
                trainingMetrics: trainingMetrics,
                validationMetrics: validationMetrics,
                modelParameters: modelParameters,
                scenePrintRevision: scenePrintRevision,
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
            moduleOutputPath: outputDirPath
        )
        print("📁 出力ディレクトリ作成成功: \(outputDirectoryURL.path)")
        return outputDirectoryURL
    }

    public func getClassLabelDirectories() throws -> [URL] {
        let classLabelDirURLs = try fileManager.getClassLabelDirectories(resourcesPath: resourcesDirectoryPath)
        print("📁 検出されたクラスラベルディレクトリ: \(classLabelDirURLs.map(\.lastPathComponent).joined(separator: ", "))")

        guard classLabelDirURLs.count >= 2 else {
            throw NSError(domain: "OvOClassifier", code: -1, userInfo: [
                NSLocalizedDescriptionKey: "OvO分類には少なくとも2つのクラスラベルディレクトリが必要です。現在 \(classLabelDirURLs.count)個。",
            ])
        }

        return classLabelDirURLs
    }

    public func prepareTrainingData(from classLabelDirURLs: [URL]) throws -> MLImageClassifier.DataSource {
        let trainingDataParentDirURL = classLabelDirURLs[0].deletingLastPathComponent()
        print("📁 トレーニングデータ親ディレクトリ: \(trainingDataParentDirURL.path)")
        return MLImageClassifier.DataSource.labeledDirectories(at: trainingDataParentDirURL)
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
        modelParameters: CreateML.MLImageClassifier.ModelParameters,
        scenePrintRevision: Int?
    ) -> MLModelMetadata {
        let augmentationFinalDescription = if !modelParameters.augmentationOptions.isEmpty {
            String(describing: modelParameters.augmentationOptions)
        } else {
            "なし"
        }

        let featureExtractorDescription = String(describing: modelParameters.featureExtractor)
        let featureExtractorDesc: String = if let revision = scenePrintRevision {
            "\(featureExtractorDescription)(revision: \(revision))"
        } else {
            featureExtractorDescription
        }

        return MLModelMetadata(
            author: author,
            shortDescription: """
            クラス: \(classLabelDirURLs.map(\.lastPathComponent).joined(separator: ", "))
            訓練正解率: \(String(format: "%.1f%%", (1.0 - trainingMetrics.classificationError) * 100.0))
            検証正解率: \(String(format: "%.1f%%", (1.0 - validationMetrics.classificationError) * 100.0))
            データ拡張: \(augmentationFinalDescription)
            特徴抽出器: \(featureExtractorDesc)
            """,
            version: version
        )
    }

    public func saveModel(
        imageClassifier: MLImageClassifier,
        modelName: String,
        modelFileName: String,
        version: String,
        outputDirectoryURL: URL,
        metadata: MLModelMetadata
    ) throws -> String {
        let modelFileURL = outputDirectoryURL.appendingPathComponent(modelFileName)

        print("💾 モデルファイル保存中: \(modelFileURL.path)")
        try imageClassifier.write(to: modelFileURL, metadata: metadata)
        print("✅ モデルファイル保存完了")

        return modelFileURL.path
    }

    public func createTrainingResult(
        modelName: String,
        classLabelDirURLs: [URL],
        trainingMetrics: MLClassifierMetrics,
        validationMetrics: MLClassifierMetrics,
        modelParameters: CreateML.MLImageClassifier.ModelParameters,
        scenePrintRevision: Int?,
        trainingDurationSeconds: TimeInterval,
        modelFilePath: String
    ) -> OvOTrainingResult {
        let augmentationFinalDescription = if !modelParameters.augmentationOptions.isEmpty {
            String(describing: modelParameters.augmentationOptions)
        } else {
            "なし"
        }

        let featureExtractorDescription = String(describing: modelParameters.featureExtractor)
        let featureExtractorDesc: String = if let revision = scenePrintRevision {
            "\(featureExtractorDescription)(revision: \(revision))"
        } else {
            featureExtractorDescription
        }

        let metadata = CICTrainingMetadata(
            modelName: modelName,
            trainingDurationInSeconds: trainingDurationSeconds,
            trainedModelFilePath: modelFilePath,
            sourceTrainingDataDirectoryPath: classLabelDirURLs[0].deletingLastPathComponent().path,
            detectedClassLabelsList: classLabelDirURLs.map(\.lastPathComponent),
            maxIterations: modelParameters.maxIterations,
            dataAugmentationDescription: augmentationFinalDescription,
            featureExtractorDescription: featureExtractorDesc
        )

        let confusionMatrix = CICMultiClassConfusionMatrix(
            dataTable: validationMetrics.confusion,
            predictedColumn: "Predicted",
            actualColumn: "True Label"
        )

        return OvOTrainingResult(
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

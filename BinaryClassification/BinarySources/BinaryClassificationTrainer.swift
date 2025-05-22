import CoreML
import CreateML
import CSInterface
import Foundation
import CSConfusionMatrix

public class BinaryClassificationTrainer: ScreeningTrainerProtocol {
    public typealias TrainingResultType = BinaryTrainingResult

    private let resourcesDirectoryPathOverride: String?
    private let outputDirectoryPathOverride: String?

    public var outputDirPath: String {
        if let overridePath = outputDirectoryPathOverride {
            return overridePath
        }
        var dir = URL(fileURLWithPath: #filePath)
        dir.deleteLastPathComponent()
        dir.deleteLastPathComponent()
        return dir.appendingPathComponent("OutputModels").path
    }

    public var classificationMethod: String { "Binary" }

    public var resourcesDirectoryPath: String {
        if let overridePath = resourcesDirectoryPathOverride {
            return overridePath
        }
        var dir = URL(fileURLWithPath: #filePath)
        dir.deleteLastPathComponent()
        dir.deleteLastPathComponent()
        return dir.appendingPathComponent("Resources").path
    }

    public init(resourcesDirectoryPathOverride: String? = nil, outputDirectoryPathOverride: String? = nil) {
        self.resourcesDirectoryPathOverride = resourcesDirectoryPathOverride
        self.outputDirectoryPathOverride = outputDirectoryPathOverride
    }

    public func train(
        author: String,
        modelName: String,
        version: String,
        modelParameters: CreateML.MLImageClassifier.ModelParameters,
        scenePrintRevision: Int?
    ) async -> BinaryTrainingResult? {
        let resourcesPath = resourcesDirectoryPath
        let resourcesDirURL = URL(fileURLWithPath: resourcesPath)

        // 出力ディレクトリ設定
        let outputDirectoryURL: URL
        do {
            outputDirectoryURL = try createOutputDirectory(
                modelName: modelName,
                version: version
            )
        } catch {
            print("❌ エラー: 出力ディレクトリ設定に失敗 \(error.localizedDescription)")
            return nil
        }

        print("🚀 Binaryトレーニング開始 (バージョン: \(version))...")

        let classLabelDirURLs: [URL]
        do {
            classLabelDirURLs = try FileManager.default.contentsOfDirectory(
                at: resourcesDirURL,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: .skipsHiddenFiles
            ).filter { url in
                var isDirectory: ObjCBool = false
                FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory)
                return isDirectory.boolValue && !url.lastPathComponent.hasPrefix(".")
            }
        } catch {
            print("🛑 エラー: リソースディレクトリ内ラベルディレクトリ取得失敗: \(error.localizedDescription)")
            return nil
        }

        guard classLabelDirURLs.count == 2 else {
            print("🛑 エラー: Binary分類には2つのクラスラベルディレクトリが必要です。現在 \(classLabelDirURLs.count)個。処理中止。")
            return nil
        }

        let trainingDataParentDirURL = classLabelDirURLs[0].deletingLastPathComponent()
        let trainingDataSource = MLImageClassifier.DataSource.labeledDirectories(at: trainingDataParentDirURL)

        do {
            let trainingStartTime = Date()
            let imageClassifier = try MLImageClassifier(trainingData: trainingDataSource, parameters: modelParameters)
            let trainingEndTime = Date()
            let trainingDurationSeconds = trainingEndTime.timeIntervalSince(trainingStartTime)

            let trainingMetrics = imageClassifier.trainingMetrics
            let validationMetrics = imageClassifier.validationMetrics

            let trainingAccuracyPercentage = (1.0 - trainingMetrics.classificationError) * 100.0
            let validationAccuracyPercentage = (1.0 - validationMetrics.classificationError) * 100.0

            // トレーニング完了後のパフォーマンス指標を表示
            print("\n📊 トレーニング結果サマリー")
            print(String(format: "  訓練正解率: %.1f%%, 検証正解率: %.1f%%",
                trainingAccuracyPercentage,
                validationAccuracyPercentage))

            // 混同行列の計算をCSBinaryConfusionMatrixに委任
            if let confusionMatrix = CSBinaryConfusionMatrix(
                dataTable: validationMetrics.confusion,
                predictedColumn: "predictedLabel",
                actualColumn: "trueLabel"
            ) {
                // 混同行列の表示
                confusionMatrix.printMatrix()
            } else {
                print("⚠️ 警告: 検証データが不十分なため、混同行列の計算をスキップしました")
            }

            // データ拡張の説明
            let augmentationFinalDescription: String
            if !modelParameters.augmentationOptions.isEmpty {
                augmentationFinalDescription = String(describing: modelParameters.augmentationOptions)
            } else {
                augmentationFinalDescription = "なし"
            }

            // 特徴抽出器の説明
            let baseFeatureExtractorString = String(describing: modelParameters.featureExtractor)
            var featureExtractorDesc: String
            if let revision = scenePrintRevision {
                featureExtractorDesc = "\(baseFeatureExtractorString)(revision: \(revision))"
            } else {
                featureExtractorDesc = baseFeatureExtractorString
            }

            // モデルのメタデータを作成
            let modelMetadata = MLModelMetadata(
                author: author,
                shortDescription: """
                クラス: \(classLabelDirURLs.map(\.lastPathComponent).joined(separator: ", "))
                訓練正解率: \(String(format: "%.1f%%", trainingAccuracyPercentage))
                検証正解率: \(String(format: "%.1f%%", validationAccuracyPercentage))
                データ拡張: \(augmentationFinalDescription)
                特徴抽出器: \(featureExtractorDesc)
                """,
                version: version
            )

            let outputModelFileURL = outputDirectoryURL
                .appendingPathComponent("\(modelName)_\(classificationMethod)_\(version).mlmodel")

            try imageClassifier.write(to: outputModelFileURL, metadata: modelMetadata)

            return BinaryTrainingResult(
                modelName: modelName,
                trainingDataAccuracyPercentage: trainingAccuracyPercentage,
                validationDataAccuracyPercentage: validationAccuracyPercentage,
                trainingDataMisclassificationRate: trainingMetrics.classificationError,
                validationDataMisclassificationRate: validationMetrics.classificationError,
                trainingDurationInSeconds: trainingDurationSeconds,
                trainedModelFilePath: outputModelFileURL.path,
                sourceTrainingDataDirectoryPath: trainingDataParentDirURL.path,
                detectedClassLabelsList: classLabelDirURLs.map(\.lastPathComponent),
                maxIterations: modelParameters.maxIterations,
                dataAugmentationDescription: augmentationFinalDescription,
                baseFeatureExtractorDescription: baseFeatureExtractorString,
                scenePrintRevision: scenePrintRevision
            )

        } catch let createMLError as CreateML.MLCreateError {
            print("🛑 エラー: モデル [\(modelName)] のトレーニングまたは保存失敗 (CreateML): \(createMLError.localizedDescription)")
            return nil
        } catch {
            print("🛑 エラー: トレーニングプロセス中に予期しないエラー: \(error.localizedDescription)")
            return nil
        }
    }
}

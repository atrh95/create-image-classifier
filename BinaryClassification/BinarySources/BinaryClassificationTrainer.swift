import CoreML
import CreateML
import CSInterface
import Foundation

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
            print("\n📊 トレーニング結果サマリー:")
            print(String(format: "  訓練正解率: %.1f%%, 検証正解率: %.1f%%",
                trainingAccuracyPercentage,
                validationAccuracyPercentage))

            var recallRate = 0.0
            var precisionRate = 0.0

            let confusionMatrix = validationMetrics.confusion
            var labelSet = Set<String>()
            for row in confusionMatrix.rows {
                if let actual = row["True Label"]?.stringValue { labelSet.insert(actual) }
                if let predicted = row["Predicted"]?.stringValue { labelSet.insert(predicted) }
            }

            if labelSet.count == 2 {
                let labels = Array(labelSet).sorted()
                let positiveLabel = labels[1]
                let negativeLabel = labels[0]

                var truePositives = 0
                var falsePositives = 0
                var falseNegatives = 0

                for row in confusionMatrix.rows {
                    guard
                        let actual = row["True Label"]?.stringValue,
                        let predicted = row["Predicted"]?.stringValue,
                        let cnt = row["Count"]?.intValue
                    else { continue }

                    if actual == positiveLabel, predicted == positiveLabel {
                        truePositives += cnt
                    } else if actual == negativeLabel, predicted == positiveLabel {
                        falsePositives += cnt
                    } else if actual == positiveLabel, predicted == negativeLabel {
                        falseNegatives += cnt
                    }
                }

                if (truePositives + falseNegatives) > 0 {
                    recallRate = Double(truePositives) / Double(truePositives + falseNegatives)
                }
                if (truePositives + falsePositives) > 0 {
                    precisionRate = Double(truePositives) / Double(truePositives + falsePositives)
                }

                print(String(format: "  陽性クラス (%@): 再現率 %.1f%%, 適合率 %.1f%%",
                    positiveLabel,
                    recallRate * 100,
                    precisionRate * 100))
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

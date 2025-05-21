import CoreML
import CreateML
import CSInterface
import Foundation

public class MultiClassClassificationTrainer: ScreeningTrainerProtocol {
    public typealias TrainingResultType = MultiClassTrainingResult

    // DI 用のプロパティ
    private let resourcesDirectoryPathOverride: String?
    private let outputDirectoryPathOverride: String?
    
    // ファイルマネージャーの静的プロパティを追加
    private static let fileManager = FileManager.default

    public var outputDirPath: String {
        if let overridePath = outputDirectoryPathOverride {
            return overridePath
        }
        var dir = URL(fileURLWithPath: #filePath)
        dir.deleteLastPathComponent()
        dir.deleteLastPathComponent()
        return dir.appendingPathComponent("OutputModels").path
    }

    public var classificationMethod: String { "MultiClass" }

    public var resourcesDirectoryPath: String {
        if let overridePath = resourcesDirectoryPathOverride {
            return overridePath
        }
        var dir = URL(fileURLWithPath: #filePath)
        dir.deleteLastPathComponent()
        dir.deleteLastPathComponent()
        return dir.appendingPathComponent("Resources").path
    }

    public init(
        resourcesDirectoryPathOverride: String? = nil,
        outputDirectoryPathOverride: String? = nil
    ) {
        self.resourcesDirectoryPathOverride = resourcesDirectoryPathOverride
        self.outputDirectoryPathOverride = outputDirectoryPathOverride
    }

    public func train(
        author: String,
        modelName: String,
        version: String,
        modelParameters: CreateML.MLImageClassifier.ModelParameters,
        scenePrintRevision: Int?
    )
        async -> MultiClassTrainingResult?
    {
        let resourcesPath = resourcesDirectoryPath
        let resourcesDir = URL(fileURLWithPath: resourcesPath)
        let trainingDataParentDir = resourcesDir

        guard Self.fileManager.fileExists(atPath: trainingDataParentDir.path) else {
            print("❌ エラー: トレーニングデータ親ディレクトリが見つかりません 。 \(trainingDataParentDir.path)")
            return nil
        }

        let finalOutputDir: URL

        do {
            finalOutputDir = try createOutputDirectory(
                modelName: modelName,
                version: version
            )

            let contents = try Self.fileManager.contentsOfDirectory(
                at: trainingDataParentDir,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: .skipsHiddenFiles
            )
            let allClassDirs = contents.filter { url in
                var isDirectory: ObjCBool = false
                return Self.fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory) && isDirectory
                    .boolValue
            }
            let classLabelsFromFileSystem = allClassDirs.map(\.lastPathComponent).sorted()
            print("📚 ファイルシステムから検出されたクラスラベル: \(classLabelsFromFileSystem.joined(separator: ", "))")

            // トレーニングに使用する総サンプル数を計算
            var totalImageSamples = 0
            for classDirURL in allClassDirs {
                if let files = try? Self.fileManager.contentsOfDirectory(
                    at: classDirURL,
                    includingPropertiesForKeys: [.isRegularFileKey],
                    options: .skipsHiddenFiles
                ) {
                    totalImageSamples += files.filter { !$0.hasDirectoryPath }.count
                }
            }

            print("\n🚀 MultiClassトレーニング開始 (バージョン: \(version))...")

            let classLabelDirURLs: [URL]
            do {
                classLabelDirURLs = try Self.fileManager.contentsOfDirectory(
                    at: resourcesDir,
                    includingPropertiesForKeys: [.isDirectoryKey],
                    options: .skipsHiddenFiles
                ).filter { url in
                    var isDirectory: ObjCBool = false
                    Self.fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory)
                    return isDirectory.boolValue && !url.lastPathComponent.hasPrefix(".")
                }
            } catch {
                print("🛑 エラー: リソースディレクトリ内ラベルディレクトリ取得失敗: \(error.localizedDescription)")
                return nil
            }

            guard classLabelDirURLs.count >= 2 else {
                print("🛑 エラー: MultiClass分類には最低2つのクラスラベルディレクトリが必要です。現在 \(classLabelDirURLs.count)個。処理中止。")
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

                let confusionMatrix = validationMetrics.confusion
                var labelSet = Set<String>()
                for row in confusionMatrix.rows {
                    if let actual = row["True Label"]?.stringValue { labelSet.insert(actual) }
                    if let predicted = row["Predicted"]?.stringValue { labelSet.insert(predicted) }
                }

                let labelsFromConfusion = Array(labelSet).sorted()
                var detailedClassMetrics: [(label: String, recall: Double, precision: Double)] = []

                for label in labelsFromConfusion {
                    var truePositives = 0
                    var falsePositives = 0
                    var falseNegatives = 0

                    for row in confusionMatrix.rows {
                        guard
                            let actual = row["True Label"]?.stringValue,
                            let predicted = row["Predicted"]?.stringValue,
                            let cnt = row["Count"]?.intValue
                        else { continue }

                        if actual == label, predicted == label {
                            truePositives += cnt
                        } else if actual != label, predicted == label {
                            falsePositives += cnt
                        } else if actual == label, predicted != label {
                            falseNegatives += cnt
                        }
                    }

                    var recall = 0.0
                    var precision = 0.0

                    if (truePositives + falseNegatives) > 0 {
                        recall = Double(truePositives) / Double(truePositives + falseNegatives)
                    }
                    if (truePositives + falsePositives) > 0 {
                        precision = Double(truePositives) / Double(truePositives + falsePositives)
                    }

                    detailedClassMetrics.append((label: label, recall: recall, precision: precision))
                    print(String(format: "  %@: 再現率 %.1f%%, 適合率 %.1f%%",
                        label,
                        recall * 100,
                        precision * 100))
                }

                // マクロ平均の計算
                let macroAverageRecallRate = detailedClassMetrics.map(\.recall).reduce(0, +) / Double(detailedClassMetrics.count)
                let macroAveragePrecisionRate = detailedClassMetrics.map(\.precision).reduce(0, +) / Double(detailedClassMetrics.count)

                // データ拡張の説明
                let augmentationFinalDescription: String
                if !modelParameters.augmentationOptions.isEmpty {
                    augmentationFinalDescription = String(describing: modelParameters.augmentationOptions)
                } else {
                    augmentationFinalDescription = "なし"
                }

                // 特徴抽出器の説明
                let featureExtractorString = String(describing: modelParameters.featureExtractor)
                var featureExtractorDesc: String
                if let revision = scenePrintRevision {
                    featureExtractorDesc = "\(featureExtractorString)(revision: \(revision))"
                } else {
                    featureExtractorDesc = featureExtractorString
                }

                // モデルのメタデータを作成
                let modelMetadata = MLModelMetadata(
                    author: author,
                    shortDescription: """
                    クラス: \(classLabelsFromFileSystem.joined(separator: ", "))
                    訓練正解率: \(String(format: "%.1f%%", trainingAccuracyPercentage))
                    検証正解率: \(String(format: "%.1f%%", validationAccuracyPercentage))
                    データ拡張: \(augmentationFinalDescription)
                    特徴抽出器: \(featureExtractorDesc)
                    """,
                    version: version
                )

                let outputModelFileURL = finalOutputDir
                    .appendingPathComponent("\(modelName)_\(classificationMethod)_\(version).mlmodel")

                try imageClassifier.write(to: outputModelFileURL, metadata: modelMetadata)

                return MultiClassTrainingResult(
                    modelName: modelName,
                    trainingDataAccuracy: trainingAccuracyPercentage / 100.0,
                    validationDataAccuracy: validationAccuracyPercentage / 100.0,
                    trainingDataErrorRate: trainingMetrics.classificationError,
                    validationDataErrorRate: validationMetrics.classificationError,
                    trainingTimeInSeconds: trainingDurationSeconds,
                    modelOutputPath: outputModelFileURL.path,
                    trainingDataPath: trainingDataParentDirURL.path,
                    classLabels: classLabelsFromFileSystem,
                    maxIterations: modelParameters.maxIterations,
                    macroAverageRecall: macroAverageRecallRate,
                    macroAveragePrecision: macroAveragePrecisionRate,
                    detectedClassLabelsList: classLabelsFromFileSystem,
                    dataAugmentationDescription: augmentationFinalDescription,
                    baseFeatureExtractorDescription: featureExtractorString,
                    scenePrintRevision: scenePrintRevision
                )

            } catch let createMLError as CreateML.MLCreateError {
                print("🛑 エラー: モデル [\(modelName)] のトレーニングまたは保存失敗 (CreateML): \(createMLError.localizedDescription)")
                return nil
            } catch {
                print("🛑 エラー: トレーニングプロセス中に予期しないエラー: \(error.localizedDescription)")
                return nil
            }

        } catch let error as CreateML.MLCreateError {
            print("  ❌ モデル [\(modelName)] のトレーニングまたは保存エラー 。CreateMLエラー: \(error.localizedDescription)")
            return nil
        } catch {
            print("  ❌ トレーニングプロセス中に予期しないエラーが発生しました 。 \(error.localizedDescription)")
            if let nsError = error as NSError? {
                print("    詳細なエラー情報: \(nsError.userInfo)")
            }
            return nil
        }
    }
}

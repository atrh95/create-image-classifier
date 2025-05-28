import CoreML
import CreateML
import CICConfusionMatrix
import CICInterface
import CICFileManager
import Foundation

public class MultiClassClassificationTrainer: ScreeningTrainerProtocol {
    public typealias TrainingResultType = MultiClassTrainingResult

    // DI 用のプロパティ
    private let resourcesDirectoryPathOverride: String?
    private let outputDirectoryPathOverride: String?
    private let fileManager: CICFileManager

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
        outputDirectoryPathOverride: String? = nil,
        fileManager: CICFileManager = CICFileManager()
    ) {
        self.resourcesDirectoryPathOverride = resourcesDirectoryPathOverride
        self.outputDirectoryPathOverride = outputDirectoryPathOverride
        self.fileManager = fileManager
    }

    public func train(
        author: String,
        modelName: String,
        version: String,
        modelParameters: CreateML.MLImageClassifier.ModelParameters,
        scenePrintRevision: Int?
    ) async -> MultiClassTrainingResult? {
        let resourcesPath = resourcesDirectoryPath
        let resourcesDir = URL(fileURLWithPath: resourcesPath)
        let trainingDataParentDir = resourcesDir

        guard FileManager.default.fileExists(atPath: trainingDataParentDir.path) else {
            print("❌ エラー: トレーニングデータ親ディレクトリが見つかりません 。 \(trainingDataParentDir.path)")
            return nil
        }

        let finalOutputDir: URL

        do {
            finalOutputDir = try fileManager.createOutputDirectory(
                modelName: modelName,
                version: version,
                classificationMethod: classificationMethod,
                moduleOutputPath: outputDirPath
            )

            let classLabelDirURLs: [URL]
            do {
                classLabelDirURLs = try fileManager.getClassLabelDirectories(resourcesPath: resourcesPath)
                print("📁 検出されたクラスラベルディレクトリ: \(classLabelDirURLs.map(\.lastPathComponent).joined(separator: ", "))")
            } catch {
                print("🛑 エラー: リソースディレクトリ内ラベルディレクトリ取得失敗: \(error.localizedDescription)")
                return nil
            }

            guard classLabelDirURLs.count >= 2 else {
                print("🛑 エラー: MultiClass分類には最低2つのクラスラベルディレクトリが必要です。現在 \(classLabelDirURLs.count)個。処理中止。")
                return nil
            }

            let classLabelsFromFileSystem = classLabelDirURLs.map(\.lastPathComponent).sorted()
            print("📚 ファイルシステムから検出されたクラスラベル: \(classLabelsFromFileSystem.joined(separator: ", "))")

            // トレーニングに使用する総サンプル数を計算
            var totalImageSamples = 0
            for classDirURL in classLabelDirURLs {
                if let files = try? fileManager.getFilesInDirectory(classDirURL) {
                    totalImageSamples += files.count
                }
            }

            print("\n🚀 MultiClassトレーニング開始 (バージョン: \(version))...")

            let trainingDataParentDirURL = classLabelDirURLs[0].deletingLastPathComponent()
            let trainingDataSource = MLImageClassifier.DataSource.labeledDirectories(at: trainingDataParentDirURL)

            do {
                let trainingStartTime = Date()
                let imageClassifier = try MLImageClassifier(
                    trainingData: trainingDataSource,
                    parameters: modelParameters
                )
                let trainingEndTime = Date()
                let trainingDurationSeconds = trainingEndTime.timeIntervalSince(trainingStartTime)

                let trainingMetrics = imageClassifier.trainingMetrics
                let validationMetrics = imageClassifier.validationMetrics

                let trainingAccuracyPercentage = (1.0 - trainingMetrics.classificationError) * 100.0
                let validationAccuracyPercentage = (1.0 - validationMetrics.classificationError) * 100.0

                // データ拡張の説明
                let commonDataAugmentationDesc = if !modelParameters.augmentationOptions.isEmpty {
                    String(describing: modelParameters.augmentationOptions)
                } else {
                    "なし"
                }

                // 特徴抽出器の説明
                let baseFeatureExtractorString = String(describing: modelParameters.featureExtractor)
                let commonFeatureExtractorDesc: String = if let revision = scenePrintRevision {
                    "\(baseFeatureExtractorString)(revision: \(revision))"
                } else {
                    baseFeatureExtractorString
                }

                // トレーニング完了後のパフォーマンス指標を表示
                print("\n📊 トレーニング結果サマリー")
                print(String(
                    format: "  訓練正解率: %.1f%%, 検証正解率: %.1f%%",
                    trainingAccuracyPercentage,
                    validationAccuracyPercentage
                ))

                // 混同行列の計算をCSMultiClassConfusionMatrixに委任
                let confusionMatrix = CSMultiClassConfusionMatrix(
                    dataTable: validationMetrics.confusion,
                    predictedColumn: "Predicted",
                    actualColumn: "True Label"
                )

                if let confusionMatrix {
                    // 混同行列の表示
                    print("\n📊 混同行列")
                    print(confusionMatrix.getMatrixGraph())

                    // 各クラスの性能指標を表示
                    print("\n📊 クラス別性能指標")
                    for metric in confusionMatrix.calculateMetrics() {
                        print(String(
                            format: "  %@: 再現率 %.1f%%, 適合率 %.1f%%, F1スコア %.1f%%",
                            metric.label,
                            metric.recall * 100.0,
                            metric.precision * 100.0,
                            metric.f1Score * 100.0
                        ))
                    }
                } else {
                    print("⚠️ 警告: 検証データが不十分なため、混同行列の計算をスキップしました")
                }

                // モデルのメタデータを作成
                let modelMetadata = MLModelMetadata(
                    author: author,
                    shortDescription: """
                    クラス: \(classLabelDirURLs.map(\.lastPathComponent).joined(separator: ", "))
                    訓練正解率: \(String(format: "%.1f%%", (1.0 - trainingMetrics.classificationError) * 100.0))
                    検証正解率: \(String(format: "%.1f%%", (1.0 - validationMetrics.classificationError) * 100.0))
                    \(confusionMatrix.map { matrix in
                        matrix.calculateMetrics().map { metric in
                            """
                            【\(metric.label)】
                            再現率: \(String(format: "%.1f%%", metric.recall * 100.0)), \
                            適合率: \(String(format: "%.1f%%", metric.precision * 100.0)), \
                            F1スコア: \(String(format: "%.1f%%", metric.f1Score * 100.0))
                            """
                        }.joined(separator: "\n")
                    } ?? "")
                    データ拡張: \(commonDataAugmentationDesc)
                    特徴抽出器: \(commonFeatureExtractorDesc)
                    """,
                    version: version
                )

                let modelFileName = "\(modelName)_\(classificationMethod)_\(version).mlmodel"
                let modelFilePath = finalOutputDir.appendingPathComponent(modelFileName).path

                try imageClassifier.write(to: URL(fileURLWithPath: modelFilePath), metadata: modelMetadata)

                return MultiClassTrainingResult(
                    modelName: modelName,
                    modelOutputPath: modelFilePath,
                    trainingDataPath: trainingDataParentDirURL.path,
                    classLabels: classLabelsFromFileSystem,
                    maxIterations: modelParameters.maxIterations,
                    dataAugmentationDescription: commonDataAugmentationDesc,
                    featureExtractorDescription: commonFeatureExtractorDesc,
                    trainingMetrics: (
                        accuracy: 1.0 - trainingMetrics.classificationError,
                        errorRate: trainingMetrics.classificationError
                    ),
                    validationMetrics: (
                        accuracy: 1.0 - validationMetrics.classificationError,
                        errorRate: validationMetrics.classificationError
                    ),
                    trainingTimeInSeconds: trainingDurationSeconds,
                    confusionMatrix: confusionMatrix
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
                print("  - エラーコード: \(nsError.code)")
                print("  - エラードメイン: \(nsError.domain)")
                print("  - エラー説明: \(nsError.localizedDescription)")
            }
            return nil
        }
    }
}

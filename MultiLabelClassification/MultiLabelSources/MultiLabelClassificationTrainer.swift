import CICConfusionMatrix
import CICFileManager
import CICInterface
import CICTrainingResult
import CoreML
import CreateML
import Foundation

private struct ImageAnnotation: Codable {
    let filename: String
    let annotations: [String]
}

public class MultiLabelClassificationTrainer: ScreeningTrainerProtocol {
    public typealias TrainingResultType = MultiLabelTrainingResult

    // DI 用のプロパティ
    private let resourcesDirectoryPathOverride: String?
    private let outputDirectoryPathOverride: String?
    private let annotationFilePathOverride: String?
    private let fileManager: CICFileManager

    public var outputDirPath: String {
        if let overridePath = outputDirectoryPathOverride {
            return overridePath
        }
        var dir = URL(fileURLWithPath: #filePath)
        dir.deleteLastPathComponent()
        dir.deleteLastPathComponent()
        return dir.appendingPathComponent("OutputModels").path
    }

    public var classificationMethod: String { "MultiLabel" }

    public var resourcesDirectoryPath: String {
        if let overridePath = resourcesDirectoryPathOverride {
            return overridePath
        }
        var dir = URL(fileURLWithPath: #filePath)
        dir.deleteLastPathComponent()
        dir.deleteLastPathComponent()
        return dir.appendingPathComponent("Resources").path
    }

    public var annotationFilePath: String? {
        if let overridePath = annotationFilePathOverride {
            return overridePath
        }
        var dir = URL(fileURLWithPath: #filePath)
        dir.deleteLastPathComponent()
        dir.deleteLastPathComponent()
        let resourcesDir = dir.appendingPathComponent("Resources")

        // Resourcesディレクトリ内のJSONファイルを探す
        guard let files = try? FileManager.default
            .contentsOfDirectory(at: resourcesDir, includingPropertiesForKeys: nil)
        else {
            return nil
        }

        // 最初に見つかったJSONファイルのパスを返す
        return files.first { $0.pathExtension.lowercased() == "json" }?.path
    }

    public init(
        resourcesDirectoryPathOverride: String? = nil,
        outputDirectoryPathOverride: String? = nil,
        annotationFilePathOverride: String? = nil,
        fileManager: CICFileManager = CICFileManager()
    ) {
        self.resourcesDirectoryPathOverride = resourcesDirectoryPathOverride
        self.outputDirectoryPathOverride = outputDirectoryPathOverride
        self.annotationFilePathOverride = annotationFilePathOverride
        self.fileManager = fileManager
    }

    public func train(
        author: String,
        modelName: String,
        version: String,
        modelParameters: CreateML.MLImageClassifier.ModelParameters,
        scenePrintRevision: Int?
    ) async -> MultiLabelTrainingResult? {
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
                print("🛑 エラー: MultiLabel分類には最低2つのクラスラベルディレクトリが必要です。現在 \(classLabelDirURLs.count)個。処理中止。")
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

            print("\n🚀 MultiLabelトレーニング開始 (バージョン: \(version))...")

            let trainingDataParentDirURL = classLabelDirURLs[0].deletingLastPathComponent()
            let trainingDataSource: MLImageClassifier.DataSource

            if let annotationPath = annotationFilePath {
                do {
                    let annotationData = try Data(contentsOf: URL(fileURLWithPath: annotationPath))
                    let annotations = try JSONDecoder().decode([ImageAnnotation].self, from: annotationData)

                    // 一時ディレクトリを作成
                    let tempDir = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(UUID().uuidString)
                    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)

                    // ラベルごとのディレクトリを作成
                    var labelDirs: [String: URL] = [:]
                    for annotation in annotations {
                        for label in annotation.annotations {
                            if labelDirs[label] == nil {
                                let labelDir = tempDir.appendingPathComponent(label)
                                try FileManager.default.createDirectory(at: labelDir, withIntermediateDirectories: true)
                                labelDirs[label] = labelDir
                            }
                        }
                    }

                    // 画像をコピー
                    for annotation in annotations {
                        let sourceURL = URL(fileURLWithPath: resourcesPath).appendingPathComponent(annotation.filename)
                        for label in annotation.annotations {
                            if let labelDir = labelDirs[label] {
                                try FileManager.default.copyItem(
                                    at: sourceURL,
                                    to: labelDir.appendingPathComponent(sourceURL.lastPathComponent)
                                )
                            }
                        }
                    }

                    trainingDataSource = MLImageClassifier.DataSource.labeledDirectories(at: tempDir)
                } catch {
                    print("❌ エラー: アノテーションファイルの読み込みに失敗しました: \(error.localizedDescription)")
                    return nil
                }
            } else {
                trainingDataSource = MLImageClassifier.DataSource.labeledDirectories(at: trainingDataParentDirURL)
            }

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

                // モデルのメタデータを作成
                let modelMetadata = MLModelMetadata(
                    author: author,
                    shortDescription: """
                    クラス: \(classLabelDirURLs.map(\.lastPathComponent).joined(separator: ", "))
                    訓練正解率: \(String(format: "%.1f%%", (1.0 - trainingMetrics.classificationError) * 100.0))
                    検証正解率: \(String(format: "%.1f%%", (1.0 - validationMetrics.classificationError) * 100.0))
                    データ拡張: \(commonDataAugmentationDesc)
                    特徴抽出器: \(commonFeatureExtractorDesc)
                    """,
                    version: version
                )

                let modelFileName = "\(modelName)_\(classificationMethod)_\(version).mlmodel"
                let modelFilePath = finalOutputDir.appendingPathComponent(modelFileName).path

                try imageClassifier.write(to: URL(fileURLWithPath: modelFilePath), metadata: modelMetadata)

                let metadata = CICTrainingMetadata(
                    modelName: modelName,
                    trainingDurationInSeconds: trainingDurationSeconds,
                    trainedModelFilePath: modelFilePath,
                    sourceTrainingDataDirectoryPath: trainingDataParentDirURL.path,
                    detectedClassLabelsList: classLabelsFromFileSystem,
                    maxIterations: modelParameters.maxIterations,
                    dataAugmentationDescription: commonDataAugmentationDesc,
                    featureExtractorDescription: commonFeatureExtractorDesc
                )

                let individualModelReports = classLabelsFromFileSystem.map { label in
                    CICIndividualModelReport(
                        modelName: modelName,
                        positiveClassName: label,
                        trainingAccuracyRate: 1.0 - trainingMetrics.classificationError,
                        validationAccuracyPercentage: 1.0 - validationMetrics.classificationError,
                        confusionMatrix: CICBinaryConfusionMatrix(
                            dataTable: validationMetrics.confusion,
                            predictedColumn: "Predicted",
                            actualColumn: "True Label",
                            positiveClass: label
                        )
                    )
                }

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
                    confusionMatrix: nil,
                    individualModelReports: individualModelReports
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

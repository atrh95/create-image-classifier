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
            .deletingLastPathComponent() // Classifiers
            .deletingLastPathComponent() // Project root
            .appendingPathComponent("CICResources")
            .appendingPathComponent("OvOResources")
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
        modelParameters: CreateML.MLImageClassifier.ModelParameters,
        scenePrintRevision: Int?
    ) async -> OvOTrainingResult? {
        print("📁 リソースディレクトリ: \(resourcesDirectoryPath)")
        print("🚀 OvOモデル作成開始 (バージョン: \(version))...")

        do {
            // クラスラベルディレクトリの取得
            let classLabelDirURLs = try getClassLabelDirectories()

            // クラスラベルを取得して組み合わせを生成
            let classLabels = classLabelDirURLs.map { $0.lastPathComponent }
            
            // nC2の組み合わせを生成
            var combinations: [(String, String)] = []
            for i in 0..<classLabels.count {
                for j in (i+1)..<classLabels.count {
                    combinations.append((classLabels[i], classLabels[j]))
                }
            }

            // 出力ディレクトリの設定
            let outputDirectoryURL = try setupOutputDirectory(modelName: modelName, version: version)

            // 各組み合わせに対してモデルを生成
            var modelFilePaths: [String] = []
            var individualModelReports: [CICIndividualModelReport] = []
            var totalTrainingDuration: TimeInterval = 0
            var firstModelTrainingMetrics: MLClassifierMetrics?
            var firstModelValidationMetrics: MLClassifierMetrics?

            for (class1, class2) in combinations {
                print("🔄 クラス組み合わせ [\(class1) vs \(class2)] のモデル作成開始...")

                // 2クラスのデータセットを準備
                let twoClassDataSource = try prepareTwoClassTrainingData(
                    class1: class1,
                    class2: class2,
                    basePath: resourcesDirectoryPath
                )

                // 2クラス用のモデルを訓練
                let (imageClassifier, trainingDurationSeconds) = try trainModel(
                    trainingDataSource: twoClassDataSource,
                    modelParameters: modelParameters
                )

                let currentTrainingMetrics = imageClassifier.trainingMetrics
                let currentValidationMetrics = imageClassifier.validationMetrics

                // 最初のモデルのメトリクスを保存
                if firstModelTrainingMetrics == nil {
                    firstModelTrainingMetrics = currentTrainingMetrics
                    firstModelValidationMetrics = currentValidationMetrics
                }

                // 2クラス用のメタデータを作成
                let twoClassMetadata = createModelMetadata(
                    author: author,
                    version: version,
                    classLabelDirURLs: [
                        URL(fileURLWithPath: resourcesDirectoryPath).appendingPathComponent(class1),
                        URL(fileURLWithPath: resourcesDirectoryPath).appendingPathComponent(class2)
                    ],
                    trainingMetrics: currentTrainingMetrics,
                    validationMetrics: currentValidationMetrics,
                    modelParameters: modelParameters
                )

                // モデルファイルを保存
                let modelFileName = "\(modelName)_\(classificationMethod)_\(class1)_vs_\(class2)_\(version).mlmodel"
                let modelFilePath = try saveMLModel(
                    imageClassifier: imageClassifier,
                    modelName: modelName,
                    modelFileName: modelFileName,
                    version: version,
                    outputDirectoryURL: outputDirectoryURL,
                    metadata: twoClassMetadata
                )

                // 個別モデルのレポートを作成
                let confusionMatrix = CICBinaryConfusionMatrix(
                    dataTable: currentValidationMetrics.confusion,
                    predictedColumn: "Predicted",
                    actualColumn: "True Label",
                    positiveClass: class2
                )

                let report = CICIndividualModelReport(
                    modelName: modelName,
                    positiveClassName: class2,
                    trainingAccuracyRate: 1.0 - currentTrainingMetrics.classificationError,
                    validationAccuracyPercentage: 1.0 - currentValidationMetrics.classificationError,
                    confusionMatrix: confusionMatrix
                )

                modelFilePaths.append(modelFilePath)
                individualModelReports.append(report)
                totalTrainingDuration += trainingDurationSeconds

                print("✅ クラス組み合わせ [\(class1) vs \(class2)] のモデル作成完了")
            }

            // トレーニング結果の表示
            print("\n📊 トレーニング結果サマリー")
            if let firstModelTrainingMetrics {
                print(String(
                    format: "  訓練正解率: %.1f%%",
                    (1.0 - firstModelTrainingMetrics.classificationError) * 100.0
                ))
            }
            if let firstModelValidationMetrics {
                print(String(
                    format: "  検証正解率: %.1f%%",
                    (1.0 - firstModelValidationMetrics.classificationError) * 100.0
                ))
            }

            // 最初のモデルファイルパスを返す（後方互換性のため）
            let modelFilePath = modelFilePaths[0]

            return createTrainingResult(
                modelName: modelName,
                classLabelDirURLs: classLabelDirURLs,
                trainingMetrics: firstModelTrainingMetrics!,
                validationMetrics: firstModelValidationMetrics!,
                modelParameters: modelParameters,
                trainingDurationSeconds: totalTrainingDuration,
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
        modelParameters: CreateML.MLImageClassifier.ModelParameters
    ) -> MLModelMetadata {
        let augmentationFinalDescription = if !modelParameters.augmentationOptions.isEmpty {
            String(describing: modelParameters.augmentationOptions)
        } else {
            "なし"
        }

        let featureExtractorDescription = String(describing: modelParameters.featureExtractor)

        return MLModelMetadata(
            author: author,
            shortDescription: """
            クラス: \(classLabelDirURLs.map(\.lastPathComponent).joined(separator: ", "))
            訓練正解率: \(String(format: "%.1f%%", (1.0 - trainingMetrics.classificationError) * 100.0))
            検証正解率: \(String(format: "%.1f%%", (1.0 - validationMetrics.classificationError) * 100.0))
            データ拡張: \(augmentationFinalDescription)
            特徴抽出器: \(featureExtractorDescription)
            """,
            version: version
        )
    }

    public func saveMLModel(
        imageClassifier: MLImageClassifier,
        modelName: String,
        modelFileName: String,
        version: String,
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
    ) -> OvOTrainingResult {
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

    private func prepareTwoClassTrainingData(class1: String, class2: String, basePath: String) throws -> MLImageClassifier.DataSource {
        let sourceDir = URL(fileURLWithPath: basePath)
        return MLImageClassifier.DataSource.labeledDirectories(at: sourceDir)
    }
}

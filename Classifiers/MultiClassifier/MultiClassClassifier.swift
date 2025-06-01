import CICConfusionMatrix
import CICFileManager
import CICInterface
import CICTrainingResult
import CoreML
import CreateML
import Foundation

public final class MultiClassClassifier: ClassifierProtocol {
    public typealias TrainingResultType = MultiClassTrainingResult

    private let fileManager = CICFileManager()
    public var outputDirectoryPathOverride: String?
    public var resourceDirPathOverride: String?

    private static let imageExtensions = Set(["jpg", "jpeg", "png"])
    private static let tempBaseDirName = "TempMultiClassTrainingData"

    public var outputParentDirPath: String {
        if let override = outputDirectoryPathOverride {
            return override
        }
        let currentFileURL = URL(fileURLWithPath: #filePath)
        return currentFileURL
            .deletingLastPathComponent() // MultiClassifier
            .deletingLastPathComponent() // Classifiers
            .deletingLastPathComponent() // Project root
            .appendingPathComponent("CICOutputModels")
            .appendingPathComponent("MultiClassClassifier")
            .path
    }

    public var resourcesDirectoryPath: String {
        if let override = resourceDirPathOverride {
            return override
        }
        let currentFileURL = URL(fileURLWithPath: #filePath)
        return currentFileURL
            .deletingLastPathComponent() // MultiClassifier
            .deletingLastPathComponent() // Classifiers
            .deletingLastPathComponent() // Project root
            .appendingPathComponent("CICResources")
            .appendingPathComponent("MultiClassResources")
            .path
    }

    public var classificationMethod: String { "MultiClass" }

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
    ) async -> MultiClassTrainingResult? {
        print("📁 リソースディレクトリ: \(resourcesDirectoryPath)")
        print("🚀 MultiClassモデル作成開始 (バージョン: \(version))...")

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
            throw NSError(domain: "MultiClassClassifier", code: -1, userInfo: [
                NSLocalizedDescriptionKey: "MultiClass分類には2つ以上のクラスラベルディレクトリが必要です。現在 \(classLabelDirURLs.count)個。",
            ])
        }

        return classLabelDirURLs
    }

    public func prepareTrainingData(from classLabelDirURLs: [URL]) throws -> MLImageClassifier.DataSource {
        // 前回値をクリア
        classImageCounts.removeAll()
        
        // 親ディレクトリを取得（最初のディレクトリの親を使用）
        guard let firstDir = classLabelDirURLs.first else {
            throw NSError(domain: "MultiClassClassifier", code: -1, userInfo: [
                NSLocalizedDescriptionKey: "クラスラベルディレクトリが空です。"
            ])
        }
        let parentDir = firstDir.deletingLastPathComponent()
        print("📁 トレーニングデータ親ディレクトリ: \(parentDir.path)")
        
        // 各クラスの画像枚数を効率的にカウント
        for classDir in classLabelDirURLs {
            let className = classDir.lastPathComponent
            let files = try FileManager.default.contentsOfDirectory(
                at: classDir,
                includingPropertiesForKeys: nil
            )
            let count = files.filter { Self.imageExtensions.contains($0.pathExtension.lowercased()) }.count
            classImageCounts[className] = count
            print("📊 \(className): \(count)枚")
        }
        
        return MLImageClassifier.DataSource.labeledDirectories(at: parentDir)
    }

    public func trainModel(
        trainingDataSource: MLImageClassifier.DataSource,
        modelParameters: CreateML.MLImageClassifier.ModelParameters
    ) throws -> (MLImageClassifier, TimeInterval) {
        let trainingStartTime = Date()
        let imageClassifier = try MLImageClassifier(trainingData: trainingDataSource, parameters: modelParameters)
        let trainingEndTime = Date()
        let trainingDurationSeconds = trainingEndTime.timeIntervalSince(trainingStartTime)
        print("✅ モデルの作成が完了 (所要時間: \(String(format: "%.1f", trainingDurationSeconds))秒)")
        return (imageClassifier, trainingDurationSeconds)
    }

    private var classImageCounts: [String: Int] = [:]
    
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
        \(classLabelDirURLs.map { "\($0.lastPathComponent): \(classImageCounts[$0.lastPathComponent] ?? 0)枚" }.joined(separator: ", "))
        訓練正解率: \(String(format: "%.1f%%", (1.0 - trainingMetrics.classificationError) * 100.0))
        検証正解率: \(String(format: "%.1f%%", (1.0 - validationMetrics.classificationError) * 100.0))
        """

        if let confusionMatrix {
            let classMetrics = confusionMatrix.calculateMetrics()
            metricsDescription += """

            クラス別性能指標:
            \(classMetrics.map { metric in
                "\(metric.label): 再現率 \(String(format: "%.1f%%", metric.recall * 100.0)), 適合率 \(String(format: "%.1f%%", metric.precision * 100.0)), F1 \(String(format: "%.3f", metric.f1Score))"
            }.joined(separator: "\n"))
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
    ) -> MultiClassTrainingResult {
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

        return MultiClassTrainingResult(
            metadata: metadata,
            trainingMetrics: (
                accuracy: 1.0 - trainingMetrics.classificationError,
                errorRate: trainingMetrics.classificationError
            ),
            validationMetrics: (
                accuracy: 1.0 - validationMetrics.classificationError,
                errorRate: validationMetrics.classificationError
            ),
            confusionMatrix: confusionMatrix
        )
    }
}

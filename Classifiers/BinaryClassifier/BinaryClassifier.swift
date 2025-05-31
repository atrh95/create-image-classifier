import CICConfusionMatrix
import CICFileManager
import CICInterface
import CICTrainingResult
import CoreML
import CreateML
import Foundation

public final class BinaryClassifier: ClassifierProtocol {
    public typealias TrainingResultType = BinaryTrainingResult

    private let fileManager: CICFileManager
    public var outputDirectoryPathOverride: String?
    public var resourceDirPathOverride: String?

    private static let imageExtensions = Set(["jpg", "jpeg", "png"])
    private static let tempBaseDirName = "TempBinaryTrainingData"

    public var outputParentDirPath: String {
        if let override = outputDirectoryPathOverride {
            return override
        }
        let currentFileURL = URL(fileURLWithPath: #filePath)
        return currentFileURL
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
            .deletingLastPathComponent() // Classifiers
            .deletingLastPathComponent() // Project root
            .appendingPathComponent("CICResources")
            .appendingPathComponent("BinaryResources")
            .path
    }

    public var classificationMethod: String { "Binary" }

    public init(
        outputDirectoryPathOverride: String? = nil,
        resourceDirPathOverride: String? = nil,
        fileManager: CICFileManager = CICFileManager()
    ) {
        self.outputDirectoryPathOverride = outputDirectoryPathOverride
        self.resourceDirPathOverride = resourceDirPathOverride
        self.fileManager = fileManager
    }

    public func create(
        author: String,
        modelName: String,
        version: String,
        modelParameters: CreateML.MLImageClassifier.ModelParameters
    ) async -> BinaryTrainingResult? {
        print("📁 リソースディレクトリ: \(resourcesDirectoryPath)")
        print("🚀 Binaryモデル作成開始 (バージョン: \(version))...")

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
            let confusionMatrix = CICBinaryConfusionMatrix(
                dataTable: validationMetrics.confusion,
                predictedColumn: "Predicted",
                actualColumn: "True Label",
                positiveClass: classLabelDirURLs[1].lastPathComponent
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
            let recall = confusionMatrix?.recall ?? 0.0
            let precision = confusionMatrix?.precision ?? 0.0
            let f1Score = confusionMatrix?.f1Score ?? 0.0
            print(
                "| \(String(format: "%14.1f%%", (1.0 - trainingMetrics.classificationError) * 100.0)) | \(String(format: "%14.1f%%", (1.0 - validationMetrics.classificationError) * 100.0)) | \(String(format: "%14.1f%%", recall * 100.0)) | \(String(format: "%14.1f%%", precision * 100.0)) | \(String(format: "%14.3f", f1Score)) |"
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

        guard classLabelDirURLs.count == 2 else {
            throw NSError(domain: "BinaryClassifier", code: -1, userInfo: [
                NSLocalizedDescriptionKey: "Binary分類には2つのクラスラベルディレクトリが必要です。現在 \(classLabelDirURLs.count)個。",
            ])
        }

        return classLabelDirURLs
    }

    public func prepareTrainingData(from _: [URL]) throws -> MLImageClassifier.DataSource {
        print("📁 トレーニングデータ親ディレクトリ: \(resourcesDirectoryPath)")
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
        let confusionMatrix = CICBinaryConfusionMatrix(
            dataTable: validationMetrics.confusion,
            predictedColumn: "Predicted",
            actualColumn: "True Label",
            positiveClass: classLabelDirURLs[1].lastPathComponent
        )

        var metricsDescription = """
        クラス: \(classLabelDirURLs.map(\.lastPathComponent).joined(separator: ", "))
        訓練正解率: \(String(format: "%.1f%%", (1.0 - trainingMetrics.classificationError) * 100.0))
        検証正解率: \(String(format: "%.1f%%", (1.0 - validationMetrics.classificationError) * 100.0))
        """

        if let confusionMatrix {
            let recall = confusionMatrix.recall
            let precision = confusionMatrix.precision
            let f1Score = confusionMatrix.f1Score
            metricsDescription += """

            再現率: \(String(format: "%.1f%%", recall * 100.0))
            適合率: \(String(format: "%.1f%%", precision * 100.0))
            F1スコア: \(String(format: "%.3f", f1Score))
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
    ) -> BinaryTrainingResult {
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

        let confusionMatrix = CICBinaryConfusionMatrix(
            dataTable: validationMetrics.confusion,
            predictedColumn: "Predicted",
            actualColumn: "True Label",
            positiveClass: classLabelDirURLs[1].lastPathComponent
        )

        let individualModelReport = CICIndividualModelReport(
            modelName: modelName,
            positiveClassName: classLabelDirURLs[1].lastPathComponent,
            negativeClassName: classLabelDirURLs[0].lastPathComponent,
            trainingAccuracyRate: 1.0 - trainingMetrics.classificationError,
            validationAccuracyRate: 1.0 - validationMetrics.classificationError,
            confusionMatrix: confusionMatrix
        )

        return BinaryTrainingResult(
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
            individualModelReport: individualModelReport
        )
    }

    public func prepareTrainingData(
        positiveClass: String,
        negativeClass: String,
        basePath: String
    ) throws -> MLImageClassifier.DataSource {
        let sourceDir = URL(fileURLWithPath: basePath)
        let positiveClassDir = sourceDir.appendingPathComponent(positiveClass)
        let negativeClassDir = sourceDir.appendingPathComponent(negativeClass)

        // 各クラスの画像ファイルを取得（ここで1回だけフィルタリング）
        let positiveClassFiles = try FileManager.default.contentsOfDirectory(
            at: positiveClassDir,
            includingPropertiesForKeys: nil
        )
        .filter { Self.imageExtensions.contains($0.pathExtension.lowercased()) }

        let negativeClassFiles = try FileManager.default.contentsOfDirectory(
            at: negativeClassDir,
            includingPropertiesForKeys: nil
        )
        .filter { Self.imageExtensions.contains($0.pathExtension.lowercased()) }

        print("📊 \(positiveClass): \(positiveClassFiles.count)枚, \(negativeClass): \(negativeClassFiles.count)枚")

        // 一時ディレクトリを準備
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(Self.tempBaseDirName)
        let tempPositiveDir = tempDir.appendingPathComponent(positiveClass)
        let tempNegativeDir = tempDir.appendingPathComponent(negativeClass)

        // 既存の一時ディレクトリをクリーンにする
        if FileManager.default.fileExists(atPath: tempDir.path) {
            try FileManager.default.removeItem(at: tempDir)
        }

        try FileManager.default.createDirectory(at: tempPositiveDir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: tempNegativeDir, withIntermediateDirectories: true)

        // 各クラスの画像をコピー
        for (index, file) in positiveClassFiles.enumerated() {
            let destination = tempPositiveDir.appendingPathComponent("\(index).\(file.pathExtension)")
            try FileManager.default.copyItem(at: file, to: destination)
        }

        for (index, file) in negativeClassFiles.enumerated() {
            let destination = tempNegativeDir.appendingPathComponent("\(index).\(file.pathExtension)")
            try FileManager.default.copyItem(at: file, to: destination)
        }

        // 一時ディレクトリからデータソースを作成
        return MLImageClassifier.DataSource.labeledDirectories(at: tempDir)
    }
}

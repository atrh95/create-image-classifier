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
    private var classImageCounts: [String: Int] = [:]

    public var outputParentDirPath: String {
        if let override = outputDirectoryPathOverride {
            return override
        }
        let currentFileURL = URL(fileURLWithPath: #filePath)
        return currentFileURL
            .deletingLastPathComponent() // MultiClassClassifier
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
            .deletingLastPathComponent() // MultiClassClassifier
            .deletingLastPathComponent() // Sources
            .deletingLastPathComponent() // MultiClassClassifier
            .appendingPathComponent("Resources")
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

    public func createAndSaveModel(
        author: String,
        modelName: String,
        version: String,
        modelParameters: CreateML.MLImageClassifier.ModelParameters,
        shouldEqualizeFileCount: Bool
    ) throws {
        print("📁 リソースディレクトリ: \(resourcesDirectoryPath)")
        print("🚀 多クラス分類モデル作成開始 (バージョン: \(version))...")

        // 出力ディレクトリの準備
        let outputDirectoryURL = try setupOutputDirectory(modelName: modelName, version: version)

        // クラスラベルディレクトリの取得
        let classLabelDirURLs = try getClassLabelDirectories()
        print("📁 検出されたクラスラベルディレクトリ: \(classLabelDirURLs.map(\.lastPathComponent).joined(separator: ", "))")

        // 各クラスの画像枚数を更新
        var classImageCounts: [String: Int] = [:]
        for classLabelDir in classLabelDirURLs {
            let files = try fileManager.contentsOfDirectory(
                at: classLabelDir,
                includingPropertiesForKeys: nil
            )
            classImageCounts[classLabelDir.lastPathComponent] = files.count
        }

        // 各クラスの画像を最小枚数に揃える
        let tempDir = fileManager.temporaryDirectory.appendingPathComponent("TempMultiClassTrainingData")
        if fileManager.fileExists(atPath: tempDir.path) {
            try? fileManager.removeItem(at: tempDir)
        }
        try fileManager.createDirectory(at: tempDir, withIntermediateDirectories: true)
        let balancedDirs = try fileManager.prepareEqualizedMinimumImageSet(
            classDirs: classLabelDirURLs,
            shouldEqualize: shouldEqualizeFileCount
        )
        // 各クラスの画像を一時ディレクトリにコピー
        for (className, dir) in balancedDirs {
            let tempClassDir = tempDir.appendingPathComponent(className)
            try fileManager.createDirectory(at: tempClassDir, withIntermediateDirectories: true)
            let files = try fileManager.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
            for file in files {
                let dest = tempClassDir.appendingPathComponent(file.lastPathComponent)
                try? fileManager.copyItem(at: file, to: dest)
            }
        }

        // 各クラスの画像枚数を更新
        for (className, dir) in balancedDirs {
            let files = try fileManager.contentsOfDirectory(
                at: dir,
                includingPropertiesForKeys: nil
            )
            classImageCounts[className] = files.count
            print("📊 \(className): \(files.count)枚")
        }

        // トレーニングデータソースを作成
        let trainingDataSource = try prepareTrainingData(
            from: classLabelDirURLs,
            shouldEqualizeFileCount: shouldEqualizeFileCount
        )

        // モデルのトレーニング
        let trainingStartTime = Date()
        let imageClassifier = try MLImageClassifier(trainingData: trainingDataSource, parameters: modelParameters)
        let trainingEndTime = Date()
        let trainingDurationSeconds = trainingEndTime.timeIntervalSince(trainingStartTime)
        print("✅ モデルの作成が完了 (所要時間: \(String(format: "%.1f", trainingDurationSeconds))秒)")

        let metrics = (
            training: imageClassifier.trainingMetrics,
            validation: imageClassifier.validationMetrics
        )

        // 混同行列の計算
        let confusionMatrix = CICMultiClassConfusionMatrix(
            dataTable: metrics.validation.confusion,
            predictedColumn: "Predicted",
            actualColumn: "True Label"
        )

        // トレーニング結果の表示
        print("\n📊 トレーニング結果サマリー")
        print(String(
            format: "  訓練正解率: %.1f%%",
            (1.0 - metrics.training.classificationError) * 100.0
        ))

        if let confusionMatrix {
            print(String(
                format: "  検証正解率: %.1f%%",
                (1.0 - metrics.validation.classificationError) * 100.0
            ))
            print(confusionMatrix.getMatrixGraph())
        } else {
            print("⚠️ 警告: 検証データが不十分なため、混同行列の計算をスキップしました")
        }

        // モデルのメタデータ作成
        let augmentationFinalDescription = if !modelParameters.augmentationOptions.isEmpty {
            String(describing: modelParameters.augmentationOptions)
        } else {
            "なし"
        }

        let featureExtractorDescription = modelParameters.algorithm.description

        let metricsDescription = createMetricsDescription(
            individualReport: CICMultiClassModelReport(
                modelFileName: "\(modelName)_\(classificationMethod)_\(version).mlmodel",
                metrics: (
                    training: (
                        accuracy: 1.0 - metrics.training.classificationError,
                        errorRate: metrics.training.classificationError
                    ),
                    validation: (
                        accuracy: 1.0 - metrics.validation.classificationError,
                        errorRate: metrics.validation.classificationError
                    )
                ),
                confusionMatrix: confusionMatrix,
                classCounts: classImageCounts
            ),
            modelParameters: modelParameters,
            augmentationFinalDescription: augmentationFinalDescription,
            featureExtractorDescription: featureExtractorDescription
        )

        let modelMetadata = MLModelMetadata(
            author: author,
            shortDescription: metricsDescription,
            version: version
        )

        // モデルファイルを保存
        let modelFileName = "\(modelName)_\(classificationMethod)_\(version).mlmodel"
        let modelFilePath = outputDirectoryURL.appendingPathComponent(modelFileName).path
        print("💾 モデルファイル保存中: \(modelFilePath)")
        try imageClassifier.write(to: URL(fileURLWithPath: modelFilePath), metadata: modelMetadata)
        print("✅ モデルファイル保存完了")

        // メタデータの作成
        let metadata = CICTrainingMetadata(
            modelName: modelName,
            classLabelCounts: classImageCounts,
            maxIterations: modelParameters.maxIterations,
            dataAugmentationDescription: augmentationFinalDescription,
            featureExtractorDescription: featureExtractorDescription
        )

        let result = MultiClassTrainingResult(
            metadata: metadata,
            metrics: (
                training: (
                    accuracy: 1.0 - metrics.training.classificationError,
                    errorRate: metrics.training.classificationError
                ),
                validation: (
                    accuracy: 1.0 - metrics.validation.classificationError,
                    errorRate: metrics.validation.classificationError
                )
            ),
            confusionMatrix: confusionMatrix
        )

        // 全モデルの比較表を表示
        result.displayComparisonTable()

        // ログを保存
        result.saveLog(
            modelAuthor: author,
            modelName: modelName,
            modelVersion: version,
            outputDirPath: outputDirectoryURL.path
        )
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

    public func prepareTrainingData(
        from classLabelDirURLs: [URL],
        shouldEqualizeFileCount: Bool
    ) throws -> MLImageClassifier.DataSource {
        print("📁 トレーニングデータ親ディレクトリ: \(resourcesDirectoryPath)")

        // 各クラスの画像を最小枚数に揃える
        let tempDir = fileManager.temporaryDirectory.appendingPathComponent("TempMultiClassTrainingData")
        if fileManager.fileExists(atPath: tempDir.path) {
            try? fileManager.removeItem(at: tempDir)
        }
        try fileManager.createDirectory(at: tempDir, withIntermediateDirectories: true)
        let balancedDirs = try fileManager.prepareEqualizedMinimumImageSet(
            classDirs: classLabelDirURLs,
            shouldEqualize: shouldEqualizeFileCount
        )
        // 各クラスの画像を一時ディレクトリにコピー
        for (className, dir) in balancedDirs {
            let tempClassDir = tempDir.appendingPathComponent(className)
            try fileManager.createDirectory(at: tempClassDir, withIntermediateDirectories: true)
            let files = try fileManager.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
            for file in files {
                let dest = tempClassDir.appendingPathComponent(file.lastPathComponent)
                try? fileManager.copyItem(at: file, to: dest)
            }
        }

        // 各クラスの画像枚数を更新
        for (className, dir) in balancedDirs {
            let files = try fileManager.contentsOfDirectory(
                at: dir,
                includingPropertiesForKeys: nil
            )
            classImageCounts[className] = files.count
            print("📊 \(className): \(files.count)枚")
        }

        // トレーニングデータソースを作成
        guard let firstClassDir = balancedDirs[classLabelDirURLs[0].lastPathComponent] else {
            throw NSError(domain: "MultiClassClassifier", code: -1, userInfo: [
                NSLocalizedDescriptionKey: "トレーニングデータの準備に失敗しました。",
            ])
        }
        return MLImageClassifier.DataSource.labeledDirectories(at: firstClassDir.deletingLastPathComponent())
    }

    private func createMetricsDescription(
        individualReport: CICMultiClassModelReport,
        modelParameters _: CreateML.MLImageClassifier.ModelParameters,
        augmentationFinalDescription: String,
        featureExtractorDescription: String
    ) -> String {
        var metricsDescription = individualReport.generateMetricsDescription()

        metricsDescription += """

        データ拡張: \(augmentationFinalDescription)
        特徴抽出器: \(featureExtractorDescription)
        """

        return metricsDescription
    }
}

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

    private static let imageExtensions = Set(["jpg", "jpeg", "png"])

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
        modelParameters: CreateML.MLImageClassifier.ModelParameters
    ) async -> OvOTrainingResult? {
        print("📁 リソースディレクトリ: \(resourcesDirectoryPath)")
        print("🚀 OvOモデル作成開始 (バージョン: \(version))...")

        do {
            // クラスラベルディレクトリの取得
            let classLabelDirURLs = try getClassLabelDirectories()

            // クラスラベルを取得して組み合わせを生成
            let classLabels = classLabelDirURLs.map(\.lastPathComponent)

            // nC2の組み合わせを生成
            var combinations: [(String, String)] = []
            for i in 0 ..< classLabels.count {
                for j in (i + 1) ..< classLabels.count {
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
                        URL(fileURLWithPath: resourcesDirectoryPath).appendingPathComponent(class2),
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
                    negativeClassName: class1,
                    trainingAccuracyRate: 1.0 - currentTrainingMetrics.classificationError,
                    validationAccuracyRate: 1.0 - currentValidationMetrics.classificationError,
                    confusionMatrix: confusionMatrix
                )

                modelFilePaths.append(modelFilePath)
                individualModelReports.append(report)
                totalTrainingDuration += trainingDurationSeconds

                print("✅ クラス組み合わせ [\(class1) vs \(class2)] のモデル作成完了")
            }

            // 全モデルの比較表を表示
            print("\n📊 全モデルの性能")
            for (index, report) in individualModelReports.enumerated() {
                print("\(index + 1). \(report.negativeClassName), \(report.positiveClassName)")
            }
            print(
                "+------------------+------------------+------------------+------------------+------------------+------------------+"
            )
            print("| No. | 訓練正解率       | 検証正解率       | 再現率           | 適合率           | F1スコア         |")
            print(
                "+-----+------------------+------------------+------------------+------------------+------------------+"
            )
            for (index, report) in individualModelReports.enumerated() {
                let recall = report.confusionMatrix?.recall ?? 0.0
                let precision = report.confusionMatrix?.precision ?? 0.0
                let f1Score = report.confusionMatrix?.f1Score ?? 0.0
                print(
                    "| \(String(format: "%2d", index + 1)) | \(String(format: "%14.1f%%", report.trainingAccuracyRate * 100.0)) | \(String(format: "%14.1f%%", report.validationAccuracyRate * 100.0)) | \(String(format: "%14.1f%%", recall * 100.0)) | \(String(format: "%14.1f%%", precision * 100.0)) | \(String(format: "%14.3f", f1Score)) |"
                )
            }
            print(
                "+-----+------------------+------------------+------------------+------------------+------------------+"
            )

            // 最初のモデルファイルパスを返す（後方互換性のため）
            let modelFilePath = modelFilePaths[0]

            // 一時ディレクトリの削除
            let tempDir = Foundation.FileManager.default.temporaryDirectory.appendingPathComponent(Self.tempBaseDirName)
            if Foundation.FileManager.default.fileExists(atPath: tempDir.path) {
                try Foundation.FileManager.default.removeItem(at: tempDir)
                print("🧹 一時ディレクトリを削除しました: \(tempDir.path)")
            }

            return createTrainingResult(
                modelName: modelName,
                classLabelDirURLs: classLabelDirURLs,
                trainingMetrics: firstModelTrainingMetrics!,
                validationMetrics: firstModelValidationMetrics!,
                modelParameters: modelParameters,
                trainingDurationSeconds: totalTrainingDuration,
                oneOfModelFilePath: modelFilePath,
                individualModelReports: individualModelReports
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
        let trainingStartTime = Date()
        let imageClassifier = try MLImageClassifier(trainingData: trainingDataSource, parameters: modelParameters)
        let trainingEndTime = Date()
        let trainingDurationSeconds = trainingEndTime.timeIntervalSince(trainingStartTime)
        print("✅ モデルの作成が完了 (所要時間: \(String(format: "%.1f", trainingDurationSeconds))秒)")
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

        // クラス名を取得
        let classLabels = classLabelDirURLs.map(\.lastPathComponent)

        // 混同行列から再現率と適合率を計算
        let confusionMatrix = CICBinaryConfusionMatrix(
            dataTable: validationMetrics.confusion,
            predictedColumn: "Predicted",
            actualColumn: "True Label",
            positiveClass: classLabels[0] // 最初のクラスを正例として扱う
        )

        // バランスを取った後の画像枚数を取得
        let sourceDir = URL(fileURLWithPath: resourcesDirectoryPath)
        let class1Dir = sourceDir.appendingPathComponent(classLabels[0])
        let class2Dir = sourceDir.appendingPathComponent(classLabels[1])

        let imageExtensions = Set(["jpg", "jpeg", "png"])

        // 各クラスの画像ファイルを取得
        let class1Files = try? FileManager.default.contentsOfDirectory(
            at: class1Dir,
            includingPropertiesForKeys: nil
        )
        .filter { imageExtensions.contains($0.pathExtension.lowercased()) }

        let class2Files = try? FileManager.default.contentsOfDirectory(
            at: class2Dir,
            includingPropertiesForKeys: nil
        )
        .filter { imageExtensions.contains($0.pathExtension.lowercased()) }

        // バランスを取った後の枚数を計算
        let class1Count = class1Files?.count ?? 0
        let class2Count = class2Files?.count ?? 0
        let balancedCount = min(class1Count, class2Count)

        var metricsDescription = """
        \(classLabels[0]): \(balancedCount)枚
        \(classLabels[1]): \(balancedCount)枚
        訓練正解率: \(String(format: "%.1f%%", (1.0 - trainingMetrics.classificationError) * 100.0))
        検証正解率: \(String(format: "%.1f%%", (1.0 - validationMetrics.classificationError) * 100.0))
        """

        if let confusionMatrix {
            var metricsText = ""
            
            if confusionMatrix.recall.isFinite {
                metricsText += "再現率: \(String(format: "%.1f%%", confusionMatrix.recall * 100.0))\n"
            }
            if confusionMatrix.precision.isFinite {
                metricsText += "適合率: \(String(format: "%.1f%%", confusionMatrix.precision * 100.0))\n"
            }
            if confusionMatrix.f1Score.isFinite {
                metricsText += "F1スコア: \(String(format: "%.3f", confusionMatrix.f1Score))"
            }

            if !metricsText.isEmpty {
                metricsDescription += "\n" + metricsText
            }
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
        trainingMetrics _: MLClassifierMetrics,
        validationMetrics _: MLClassifierMetrics,
        modelParameters: CreateML.MLImageClassifier.ModelParameters,
        trainingDurationSeconds: TimeInterval,
        oneOfModelFilePath: String,
        individualModelReports: [CICIndividualModelReport]
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
            trainedModelFilePath: oneOfModelFilePath,
            detectedClassLabelsList: classLabelDirURLs.map(\.lastPathComponent),
            maxIterations: modelParameters.maxIterations,
            dataAugmentationDescription: augmentationFinalDescription,
            featureExtractorDescription: featureExtractorDescription
        )

        return OvOTrainingResult(
            metadata: metadata,
            individualModelReports: individualModelReports
        )
    }

    public func balanceClassImages(
        class1: String,
        class2: String,
        basePath: String
    ) throws -> (class1Count: Int, class2Count: Int) {
        let sourceDir = URL(fileURLWithPath: basePath)
        let class1Dir = sourceDir.appendingPathComponent(class1)
        let class2Dir = sourceDir.appendingPathComponent(class2)

        let imageExtensions = Set(["jpg", "jpeg", "png"])

        // 各クラスの画像ファイルを取得
        let class1Files = try FileManager.default.contentsOfDirectory(at: class1Dir, includingPropertiesForKeys: nil)
            .filter { imageExtensions.contains($0.pathExtension.lowercased()) }

        let class2Files = try FileManager.default.contentsOfDirectory(at: class2Dir, includingPropertiesForKeys: nil)
            .filter { imageExtensions.contains($0.pathExtension.lowercased()) }

        // 最小枚数を取得
        let minCount = min(class1Files.count, class2Files.count)

        return (minCount, minCount)
    }

    public func prepareTwoClassTrainingData(
        class1: String,
        class2: String,
        basePath: String
    ) throws -> MLImageClassifier.DataSource {
        let sourceDir = URL(fileURLWithPath: basePath)
        let class1Dir = sourceDir.appendingPathComponent(class1)
        let class2Dir = sourceDir.appendingPathComponent(class2)

        // 各クラスの画像ファイルを取得（ここで1回だけフィルタリング）
        let class1Files = try FileManager.default.contentsOfDirectory(
            at: class1Dir,
            includingPropertiesForKeys: nil
        )
        .filter { Self.imageExtensions.contains($0.pathExtension.lowercased()) }

        let class2Files = try FileManager.default.contentsOfDirectory(
            at: class2Dir,
            includingPropertiesForKeys: nil
        )
        .filter { Self.imageExtensions.contains($0.pathExtension.lowercased()) }

        // 最小枚数を取得
        let minCount = min(class1Files.count, class2Files.count)

        // 各クラスから最小枚数分の画像をランダムに選択
        let selectedClass1Files = Array(class1Files.shuffled().prefix(minCount))
        let selectedClass2Files = Array(class2Files.shuffled().prefix(minCount))

        print("📊 \(class1): \(selectedClass1Files.count)枚, \(class2): \(selectedClass2Files.count)枚")

        // 一時ディレクトリを準備
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(Self.tempBaseDirName)
        let tempClass1Dir = tempDir.appendingPathComponent(class1)
        let tempClass2Dir = tempDir.appendingPathComponent(class2)

        // 既存の一時ディレクトリをクリーンにする
        if FileManager.default.fileExists(atPath: tempDir.path) {
            try FileManager.default.removeItem(at: tempDir)
        }

        try FileManager.default.createDirectory(at: tempClass1Dir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: tempClass2Dir, withIntermediateDirectories: true)

        // 各クラスの画像をコピー
        for (index, file) in selectedClass1Files.enumerated() {
            let destination = tempClass1Dir.appendingPathComponent("\(index).\(file.pathExtension)")
            try FileManager.default.copyItem(at: file, to: destination)
        }

        for (index, file) in selectedClass2Files.enumerated() {
            let destination = tempClass2Dir.appendingPathComponent("\(index).\(file.pathExtension)")
            try FileManager.default.copyItem(at: file, to: destination)
        }

        // 一時ディレクトリからデータソースを作成
        return MLImageClassifier.DataSource.labeledDirectories(at: tempDir)
    }
}

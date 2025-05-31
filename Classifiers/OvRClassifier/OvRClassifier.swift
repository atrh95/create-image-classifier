import CICConfusionMatrix
import CICFileManager
import CICInterface
import CICTrainingResult
import Combine
import CoreML
import CreateML
import Foundation
import TabularData

public final class OvRClassifier: ClassifierProtocol {
    public typealias TrainingResultType = OvRTrainingResult

    private let fileManager = CICFileManager()
    public var outputDirectoryPathOverride: String?
    public var resourceDirPathOverride: String?

    public var outputParentDirPath: String {
        if let override = outputDirectoryPathOverride {
            return override
        }
        let currentFileURL = URL(fileURLWithPath: #filePath)
        return currentFileURL
            .deletingLastPathComponent() // OvRClassifier
            .deletingLastPathComponent() // Classifiers
            .deletingLastPathComponent() // Project root
            .appendingPathComponent("CICOutputModels")
            .appendingPathComponent("OvRClassifier")
            .path
    }

    public var classificationMethod: String { "OvR" }

    public var resourcesDirectoryPath: String {
        if let override = resourceDirPathOverride {
            return override
        }
        let currentFileURL = URL(fileURLWithPath: #filePath)
        return currentFileURL
            .deletingLastPathComponent() // OvRClassifier
            .deletingLastPathComponent() // Classifiers
            .deletingLastPathComponent() // Project root
            .appendingPathComponent("CICResources")
            .appendingPathComponent("OvRResources")
            .path
    }

    public init(
        outputDirectoryPathOverride: String? = nil,
        resourceDirPathOverride: String? = nil
    ) {
        self.outputDirectoryPathOverride = outputDirectoryPathOverride
        self.resourceDirPathOverride = resourceDirPathOverride
    }

    static let tempBaseDirName = "TempOvRTrainingData"

    public func create(
        author: String,
        modelName: String,
        version: String,
        modelParameters: CreateML.MLImageClassifier.ModelParameters,
        scenePrintRevision: Int?
    ) async -> OvRTrainingResult? {
        print("📁 リソースディレクトリ: \(resourcesDirectoryPath)")
        print("🚀 OvRモデル作成開始 (バージョン: \(version))...")

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

            // クラスラベルを取得してファイル名を生成
            let classLabels = classLabelDirURLs.map { $0.lastPathComponent }

            // OvRの場合は、Oneのクラス名のみを使用
            let oneClassLabel = classLabels.first ?? ""
            let modelFileName = "\(modelName)_\(classificationMethod)_\(oneClassLabel)_\(version).mlmodel"

            let modelFilePath = try saveMLModel(
                imageClassifier: imageClassifier,
                modelName: modelName,
                modelFileName: modelFileName,
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
            throw NSError(domain: "OvRClassifier", code: -1, userInfo: [
                NSLocalizedDescriptionKey: "OvR分類には少なくとも2つのクラスラベルディレクトリが必要です。現在 \(classLabelDirURLs.count)個。",
            ])
        }

        return classLabelDirURLs
    }

    public func prepareTrainingData(from classLabelDirURLs: [URL]) throws -> MLImageClassifier.DataSource {
        print("📁 トレーニングデータ親ディレクトリ: \(resourcesDirectoryPath)")
        
        // 一時ディレクトリの作成
        let tempDir = Foundation.FileManager.default.temporaryDirectory.appendingPathComponent(Self.tempBaseDirName)
        if Foundation.FileManager.default.fileExists(atPath: tempDir.path) {
            try Foundation.FileManager.default.removeItem(at: tempDir)
        }
        try Foundation.FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        
        // Oneクラス（最初のクラス）のデータをコピー
        let oneClassDir = tempDir.appendingPathComponent(classLabelDirURLs[0].lastPathComponent)
        try Foundation.FileManager.default.createDirectory(at: oneClassDir, withIntermediateDirectories: true)
        try copyDirectoryContents(from: classLabelDirURLs[0], to: oneClassDir)
        
        // Oneクラスの画像枚数を取得
        let oneClassCount = try Foundation.FileManager.default.contentsOfDirectory(at: oneClassDir, includingPropertiesForKeys: nil).count
        
        // 各restクラスから取得する枚数を計算
        let restClassCount = classLabelDirURLs.count - 1
        let samplesPerRestClass = Int(ceil(Double(oneClassCount) / Double(restClassCount)))
        print("📊 Oneクラス [\(classLabelDirURLs[0].lastPathComponent)]: \(oneClassCount)枚, restクラス: \(restClassCount)個, restクラスあたり: \(samplesPerRestClass)枚, 合計rest: \(samplesPerRestClass * restClassCount)枚")
        
        // 負例クラスのディレクトリを作成
        let restDir = tempDir.appendingPathComponent("rest")
        try Foundation.FileManager.default.createDirectory(at: restDir, withIntermediateDirectories: true)
        
        // 各負例クラスからサンプリングしてコピー
        var totalRestCount = 0
        for i in 1..<classLabelDirURLs.count {
            let files = try Foundation.FileManager.default.contentsOfDirectory(
                at: classLabelDirURLs[i],
                includingPropertiesForKeys: nil
            )
            let sampledFiles = files.shuffled().prefix(samplesPerRestClass)
            
            for (index, file) in sampledFiles.enumerated() {
                let destination = restDir.appendingPathComponent("\(totalRestCount + index).\(file.pathExtension)")
                try Foundation.FileManager.default.copyItem(at: file, to: destination)
            }
            totalRestCount += sampledFiles.count
        }
        
        print("📊 合計rest枚数: \(totalRestCount)")
        
        return MLImageClassifier.DataSource.labeledDirectories(at: tempDir)
    }
    
    private func copyDirectoryContents(from source: URL, to destination: URL) throws {
        let fileManager = Foundation.FileManager.default
        let contents = try fileManager.contentsOfDirectory(at: source, includingPropertiesForKeys: nil)
        
        for file in contents {
            let destinationFile = destination.appendingPathComponent(file.lastPathComponent)
            try fileManager.copyItem(at: file, to: destinationFile)
        }
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
    ) -> OvRTrainingResult {
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

        return OvRTrainingResult(
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

    public func balanceClassImages(positiveClass: String, basePath: String) throws -> (positiveCount: Int, negativeCount: Int) {
        let sourceDir = URL(fileURLWithPath: basePath)
        let positiveDir = sourceDir.appendingPathComponent(positiveClass)
        
        // 正例クラスの画像ファイルを取得
        let positiveFiles = try FileManager.default.contentsOfDirectory(at: positiveDir, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension.lowercased() == "jpg" || $0.pathExtension.lowercased() == "jpeg" || $0.pathExtension.lowercased() == "png" }
        
        // 負例クラスの画像ファイルを取得
        var negativeFiles: [URL] = []
        let classDirs = try FileManager.default.contentsOfDirectory(at: sourceDir, includingPropertiesForKeys: [.isDirectoryKey])
            .filter { $0.lastPathComponent != positiveClass }
        
        for classDir in classDirs {
            let files = try FileManager.default.contentsOfDirectory(at: classDir, includingPropertiesForKeys: nil)
                .filter { $0.pathExtension.lowercased() == "jpg" || $0.pathExtension.lowercased() == "jpeg" || $0.pathExtension.lowercased() == "png" }
            negativeFiles.append(contentsOf: files)
        }
        
        // 正例と負例の最小枚数を取得
        let minCount = min(positiveFiles.count, negativeFiles.count)
        
        return (minCount, minCount)
    }

    public func prepareTrainingData(positiveClass: String, basePath: String) throws -> MLImageClassifier.DataSource {
        let sourceDir = URL(fileURLWithPath: basePath)
        let positiveClassDir = sourceDir.appendingPathComponent(positiveClass)
        
        // 正例クラスの画像ファイルを取得
        let positiveClassFiles = try FileManager.default.contentsOfDirectory(at: positiveClassDir, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension.lowercased() == "jpg" || $0.pathExtension.lowercased() == "jpeg" || $0.pathExtension.lowercased() == "png" }
        
        // 負例クラスの画像ファイルを取得
        var negativeClassFiles: [URL] = []
        let subdirectories = try FileManager.default.contentsOfDirectory(at: sourceDir, includingPropertiesForKeys: [.isDirectoryKey])
            .filter { $0.hasDirectoryPath && $0.lastPathComponent != positiveClass }
        
        for subdir in subdirectories {
            let files = try FileManager.default.contentsOfDirectory(at: subdir, includingPropertiesForKeys: nil)
                .filter { $0.pathExtension.lowercased() == "jpg" || $0.pathExtension.lowercased() == "jpeg" || $0.pathExtension.lowercased() == "png" }
            negativeClassFiles.append(contentsOf: files)
        }
        
        // 最小枚数を取得
        let minCount = min(positiveClassFiles.count, negativeClassFiles.count)
        
        print("📊 正例クラス [\(positiveClass)] の画像枚数: \(positiveClassFiles.count)")
        print("📊 負例クラスの画像枚数: \(negativeClassFiles.count)")
        print("📊 最小枚数に合わせて \(minCount) 枚に統一します")
        
        // 一時ディレクトリを作成
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(Self.tempBaseDirName)
        let tempPositiveDir = tempDir.appendingPathComponent(positiveClass)
        let tempNegativeDir = tempDir.appendingPathComponent("negative")
        
        // 既存の一時ディレクトリを削除
        if FileManager.default.fileExists(atPath: tempDir.path) {
            try FileManager.default.removeItem(at: tempDir)
        }
        
        // 一時ディレクトリを作成
        try FileManager.default.createDirectory(at: tempPositiveDir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: tempNegativeDir, withIntermediateDirectories: true)
        
        // ランダムに選択してコピー
        let shuffledPositiveFiles = positiveClassFiles.shuffled().prefix(minCount)
        let shuffledNegativeFiles = negativeClassFiles.shuffled().prefix(minCount)
        
        for (index, file) in shuffledPositiveFiles.enumerated() {
            let destination = tempPositiveDir.appendingPathComponent("\(index).\(file.pathExtension)")
            try FileManager.default.copyItem(at: file, to: destination)
        }
        
        for (index, file) in shuffledNegativeFiles.enumerated() {
            let destination = tempNegativeDir.appendingPathComponent("\(index).\(file.pathExtension)")
            try FileManager.default.copyItem(at: file, to: destination)
        }
        
        // 一時ディレクトリからデータソースを作成
        return MLImageClassifier.DataSource.labeledDirectories(at: tempDir)
    }
}

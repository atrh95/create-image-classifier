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
            print("📁 検出されたクラスラベルディレクトリ: \(classLabelDirURLs.map(\.lastPathComponent).joined(separator: ", "))")

            var individualModelReports: [CICIndividualModelReport] = []
            var firstModelTrainingMetrics: MLClassifierMetrics?
            var firstModelValidationMetrics: MLClassifierMetrics?
            var modelFilePaths: [String] = []

            // 出力ディレクトリの設定（最初に1回だけ作成）
            let outputDirectoryURL = try setupOutputDirectory(modelName: modelName, version: version)

            // 各クラスに対して1つのモデルを作成
            for (index, oneClassDir) in classLabelDirURLs.enumerated() {
                let oneClassLabel = oneClassDir.lastPathComponent
                print("🔄 クラス [\(oneClassLabel)] のモデル作成開始...")

                // トレーニングデータの準備
                let trainingDataSource = try prepareTrainingData(positiveClass: oneClassLabel, basePath: resourcesDirectoryPath)
                print("📊 トレーニングデータソース作成完了")

                // モデルのトレーニング
                let (imageClassifier, trainingDurationSeconds) = try trainModel(
                    trainingDataSource: trainingDataSource,
                    modelParameters: modelParameters
                )

                let currentTrainingMetrics = imageClassifier.trainingMetrics
                let currentValidationMetrics = imageClassifier.validationMetrics

                // 最初のモデルのメトリクスを保存
                if firstModelTrainingMetrics == nil {
                    firstModelTrainingMetrics = currentTrainingMetrics
                    firstModelValidationMetrics = currentValidationMetrics
                }

                // モデルのメタデータ作成
                let modelMetadata = createModelMetadata(
                    author: author,
                    version: version,
                    classLabelDirURLs: [oneClassDir],
                    trainingMetrics: currentTrainingMetrics,
                    validationMetrics: currentValidationMetrics,
                    modelParameters: modelParameters
                )

                // モデルファイル名を生成
                let modelFileName = "\(modelName)_\(classificationMethod)_\(oneClassLabel)_\(version).mlmodel"

                // モデルファイルを保存
                let modelFilePath = try saveMLModel(
                    imageClassifier: imageClassifier,
                    modelName: modelName,
                    modelFileName: modelFileName,
                    version: version,
                    outputDirectoryURL: outputDirectoryURL,
                    metadata: modelMetadata
                )
                modelFilePaths.append(modelFilePath)

                // 個別モデルのレポートを作成
                let confusionMatrix = CICBinaryConfusionMatrix(
                    dataTable: currentValidationMetrics.confusion,
                    predictedColumn: "Predicted",
                    actualColumn: "True Label",
                    positiveClass: oneClassLabel
                )

                let individualReport = CICIndividualModelReport(
                    modelName: modelFileName,
                    positiveClassName: oneClassLabel,
                    trainingAccuracyRate: 1.0 - currentTrainingMetrics.classificationError,
                    validationAccuracyPercentage: 1.0 - currentValidationMetrics.classificationError,
                    confusionMatrix: confusionMatrix
                )
                individualModelReports.append(individualReport)
            }

            // 最初のモデルのメトリクスを使用してトレーニング結果を作成
            guard let firstModelTrainingMetrics = firstModelTrainingMetrics,
                  let firstModelValidationMetrics = firstModelValidationMetrics else {
                throw NSError(domain: "OvRClassifier", code: -1, userInfo: [NSLocalizedDescriptionKey: "Training failed"])
            }

            return createTrainingResult(
                modelName: modelName,
                classLabelDirURLs: classLabelDirURLs,
                trainingMetrics: firstModelTrainingMetrics,
                validationMetrics: firstModelValidationMetrics,
                modelParameters: modelParameters,
                trainingDurationSeconds: 0,
                oneOfModelFilePath: modelFilePaths[0],
                individualModelReports: individualModelReports
            )

        } catch {
            print("❌ モデル作成失敗: \(error.localizedDescription)")
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
        oneOfModelFilePath: String,
        individualModelReports: [CICIndividualModelReport]
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
            trainedModelFilePath: oneOfModelFilePath,
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
            individualModelReports: individualModelReports
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
        
        // 各restクラスから均等にサンプリング
        let samplesPerRestClass = Int(ceil(Double(positiveClassFiles.count) / Double(subdirectories.count)))
        print("📊 各restクラスから \(samplesPerRestClass) 枚ずつサンプリングします")
        
        for subdir in subdirectories {
            let files = try FileManager.default.contentsOfDirectory(at: subdir, includingPropertiesForKeys: nil)
                .filter { $0.pathExtension.lowercased() == "jpg" || $0.pathExtension.lowercased() == "jpeg" || $0.pathExtension.lowercased() == "png" }
            let sampledFiles = files.shuffled().prefix(samplesPerRestClass)
            negativeClassFiles.append(contentsOf: sampledFiles)
        }
        
        print("📊 正例クラス [\(positiveClass)] の画像枚数: \(positiveClassFiles.count)")
        print("📊 収集した負例画像枚数: \(negativeClassFiles.count)")

        // 一時ディレクトリを準備
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(Self.tempBaseDirName)
        let tempPositiveDir = tempDir.appendingPathComponent(positiveClass)
        let tempRestDir = tempDir.appendingPathComponent("rest")

        // 既存の一時ディレクトリをクリーンにする
        if FileManager.default.fileExists(atPath: tempDir.path) {
            try FileManager.default.removeItem(at: tempDir)
        }

        try FileManager.default.createDirectory(at: tempPositiveDir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: tempRestDir, withIntermediateDirectories: true)

        // 正例は全画像をコピー
        for (index, file) in positiveClassFiles.enumerated() {
            let destination = tempPositiveDir.appendingPathComponent("\(index).\(file.pathExtension)")
            try FileManager.default.copyItem(at: file, to: destination)
        }

        // 負例はサンプリング済みの画像をすべてコピー
        for (index, file) in negativeClassFiles.enumerated() {
            let destination = tempRestDir.appendingPathComponent("\(index).\(file.pathExtension)")
            try FileManager.default.copyItem(at: file, to: destination)
        }

        // 一時ディレクトリからデータソースを作成
        return MLImageClassifier.DataSource.labeledDirectories(at: tempDir)
    }
}

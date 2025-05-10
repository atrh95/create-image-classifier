import CoreML
import CreateML
import CSInterface
import Foundation

public class MultiClassClassificationTrainer: ScreeningTrainerProtocol {
    public typealias TrainingResultType = MultiClassTrainingResult

    public var modelName: String { "ScaryCatScreeningML_MultiClass" }
    public var customOutputDirPath: String { "MultiClassClassification/OutputModels" }

    public var outputRunNamePrefix: String { "MultiClass" }

    public var resourcesDirectoryPath: String {
        var dir = URL(fileURLWithPath: #filePath)
        dir.deleteLastPathComponent() // Sourcesディレクトリへ
        dir.deleteLastPathComponent() // MultiClassClassificationディレクトリへ
        return dir.appendingPathComponent("Resources").path
    }

    public init() {}

    public func train(
        author: String,
        shortDescription: String,
        version: String,
        maxIterations: Int
    )
        async -> MultiClassTrainingResult?
    {
        let resourcesPath = resourcesDirectoryPath
        let resourcesDir = URL(fileURLWithPath: resourcesPath)
        let trainingDataParentDir = resourcesDir

        guard FileManager.default.fileExists(atPath: trainingDataParentDir.path) else {
            print("❌ エラー: トレーニングデータ親ディレクトリが見つかりません: \(trainingDataParentDir.path)")
            return nil
        }

        let fileManager = FileManager.default
        let finalOutputDir: URL

        do {
            finalOutputDir = try setupVersionedRunOutputDirectory(
                version: version,
                fileManager: fileManager,
                trainerFilePath: #filePath
            )

            let contents = try fileManager.contentsOfDirectory(
                at: trainingDataParentDir,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: .skipsHiddenFiles
            )
            let allClassDirs = contents.filter { url in
                var isDirectory: ObjCBool = false
                return fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory) && isDirectory.boolValue
            }
            let classLabels = allClassDirs.map(\.lastPathComponent).sorted()
            print("📚 検出されたクラスラベル: \(classLabels.joined(separator: ", "))")

            print("\n🚀 多クラス分類モデル [\(modelName)] のトレーニングを開始します...")
            let trainingDataSource = MLImageClassifier.DataSource.labeledDirectories(at: trainingDataParentDir)

            var parameters = MLImageClassifier.ModelParameters()
            parameters.featureExtractor = .scenePrint(revision: 1)
            parameters.maxIterations = maxIterations
            parameters.validation = .split(strategy: .automatic)

            let startTime = Date()
            let model =
                try MLImageClassifier(trainingData: trainingDataSource, parameters: parameters)
            let endTime = Date()
            let duration = endTime.timeIntervalSince(startTime)
            print("🎉 [\(modelName)] のトレーニングに成功しました！ (所要時間: \(String(format: "%.2f", duration))秒)")

            let trainingEvaluation = model.trainingMetrics
            let validationEvaluation = model.validationMetrics

            let trainingDataAccuracyPercentage = (1.0 - trainingEvaluation.classificationError) * 100
            let trainingAccStr = String(format: "%.2f", trainingDataAccuracyPercentage)
            print("  📊 トレーニングデータ正解率: \(trainingAccStr)%")

            let validationDataAccuracyPercentage = (1.0 - validationEvaluation.classificationError) * 100
            let validationAccStr = String(format: "%.2f", validationDataAccuracyPercentage)
            print("  📈 検証データ正解率: \(validationAccStr)%")

            let metadata = MLModelMetadata(
                author: author,
                shortDescription: shortDescription,
                version: version
            )

            let outputModelURL = finalOutputDir.appendingPathComponent("\(modelName)_\(version).mlmodel")

            print("  💾 [\(modelName)_\(version).mlmodel] を保存中: \(outputModelURL.path)")
            try model.write(to: outputModelURL, metadata: metadata)
            print("  ✅ [\(modelName)_\(version).mlmodel] は正常に保存されました。")

            return MultiClassTrainingResult(
                modelName: modelName,
                trainingDataAccuracy: trainingDataAccuracyPercentage,
                validationDataAccuracy: validationDataAccuracyPercentage,
                trainingDataErrorRate: trainingEvaluation.classificationError,
                validationDataErrorRate: validationEvaluation.classificationError,
                trainingTimeInSeconds: duration,
                modelOutputPath: outputModelURL.path,
                trainingDataPath: trainingDataParentDir.path,
                classLabels: classLabels,
                maxIterations: maxIterations
            )

        } catch let error as CreateML.MLCreateError {
            print("  ❌ モデル [\(modelName)] のトレーニングまたは保存エラー (CreateML): \(error.localizedDescription)")
            return nil
        } catch {
            print("  ❌ トレーニングプロセス中に予期しないエラーが発生しました: \(error.localizedDescription)")
            if let nsError = error as NSError? {
                print("    詳細なエラー情報: \(nsError.userInfo)")
            }
            return nil
        }
    }
}

import CoreML
import CreateML
import CSInterface
import Foundation

public class MultiClassClassificationTrainer: ScreeningTrainerProtocol {
    public typealias TrainingResultType = MultiClassTrainingResult

    public var outputDirPath: String {
        var dir = URL(fileURLWithPath: #filePath)
        dir.deleteLastPathComponent()
        dir.deleteLastPathComponent()
        return dir.appendingPathComponent("OutputModels").path
    }

    public var classificationMethod: String { "MultiClass" }

    public var resourcesDirectoryPath: String {
        var dir = URL(fileURLWithPath: #filePath)
        dir.deleteLastPathComponent()
        dir.deleteLastPathComponent() 
        return dir.appendingPathComponent("Resources").path
    }

    public init() {}

    public func train(
        author: String,
        modelName: String,
        version: String,
        maxIterations: Int
    )
        async -> MultiClassTrainingResult?
    {
        let resourcesPath = resourcesDirectoryPath
        let resourcesDir = URL(fileURLWithPath: resourcesPath)
        let trainingDataParentDir = resourcesDir

        guard FileManager.default.fileExists(atPath: trainingDataParentDir.path) else {
            print("❌ エラー: トレーニングデータ親ディレクトリが見つかりません 。 \(trainingDataParentDir.path)")
            return nil
        }

        let finalOutputDir: URL

        do {
            finalOutputDir = try createOutputDirectory(
                modelName: modelName,
                version: version
            )

            let contents = try FileManager.default.contentsOfDirectory(
                at: trainingDataParentDir,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: .skipsHiddenFiles
            )
            let allClassDirs = contents.filter { url in
                var isDirectory: ObjCBool = false
                return FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) && isDirectory
                    .boolValue
            }
            let classLabelsFromFileSystem = allClassDirs.map(\.lastPathComponent).sorted()
            print("📚 ファイルシステムから検出されたクラスラベル: \(classLabelsFromFileSystem.joined(separator: ", "))")

            // トレーニングに使用する総サンプル数を計算
            var totalImageSamples = 0
            for classDirURL in allClassDirs {
                if let files = try? FileManager.default.contentsOfDirectory(
                    at: classDirURL,
                    includingPropertiesForKeys: [.isRegularFileKey],
                    options: .skipsHiddenFiles
                ) {
                    totalImageSamples += files.filter { !$0.hasDirectoryPath }.count // Ensure we count only files
                }
            }

            print("\n🚀 多クラス分類モデル [\(modelName)] のトレーニングを開始します...")
            let trainingDataSource = MLImageClassifier.DataSource.labeledDirectories(at: trainingDataParentDir)

            var parameters = MLImageClassifier.ModelParameters()
            parameters.featureExtractor = .scenePrint(revision: 1)
            parameters.maxIterations = maxIterations
            parameters.validation = .split(strategy: .automatic)

            let startTime = Date()
            let model = try MLImageClassifier(trainingData: trainingDataSource, parameters: parameters)
            let endTime = Date()
            let duration = endTime.timeIntervalSince(startTime)
            print("🎉 [\(modelName)] のトレーニングに成功しました！ (所要時間: \(String(format: "%.2f", duration))秒)")

            let trainingEvaluation = model.trainingMetrics
            let validationEvaluation = model.validationMetrics

            let trainingDataAccuracyPercentage = (1.0 - trainingEvaluation.classificationError) * 100
            let trainingAccuracyPercentageString = String(format: "%.2f", trainingDataAccuracyPercentage)
            print("  📊 トレーニングデータ正解率: \(trainingAccuracyPercentageString)%")

            let validationDataAccuracyPercentage = (1.0 - validationEvaluation.classificationError) * 100
            let validationAccuracyPercentageString = String(format: "%.2f", validationDataAccuracyPercentage)
            print("  📈 検証データ正解率: \(validationAccuracyPercentageString)%")

            var perClassRecallRates: [Double] = []
            var perClassPrecisionRates: [Double] = []

            let confusionMatrix = validationEvaluation.confusion
            var labelSet = Set<String>()
            for row in confusionMatrix.rows {
                if let actual = row["actualLabel"]?.stringValue {
                    labelSet.insert(actual)
                }
                if let predicted = row["predictedLabel"]?.stringValue {
                    labelSet.insert(predicted)
                }
            }
            let labelsFromConfusion = Array(labelSet).sorted()
            print("📊 混同行列から取得した評価用クラスラベル: \(labelsFromConfusion.joined(separator: ", "))")

            for label in labelsFromConfusion {
                // TP (True Positive): 真のラベルが `label` で、予測も `label`
                let truePositivesCount = confusionMatrix.rows.reduce(0.0) { acc, row in
                    guard
                        row["actualLabel"]?.stringValue == label,
                        row["predictedLabel"]?.stringValue == label,
                        let count = row["count"]?.doubleValue
                    else { return acc }
                    return acc + count
                }

                // FP (False Positive): 真のラベルは `label` 以外だが、予測は `label`
                var falsePositivesCount: Double = 0
                for row in confusionMatrix.rows {
                    guard
                        let actual = row["actualLabel"]?.stringValue,
                        let predicted = row["predictedLabel"]?.stringValue,
                        let count = row["count"]?.doubleValue,
                        actual != label, predicted == label
                    else { continue }
                    falsePositivesCount += count
                }

                // FN (False Negative): 真のラベルは `label` だが、予測は `label` 以外
                var falseNegativesCount: Double = 0
                for row in confusionMatrix.rows {
                    guard
                        let actual = row["actualLabel"]?.stringValue,
                        let predicted = row["predictedLabel"]?.stringValue,
                        let count = row["count"]?.doubleValue,
                        actual == label, predicted != label
                    else { continue }
                    falseNegativesCount += count
                }

                let recallRate = (truePositivesCount + falseNegativesCount == 0) ? 0 : truePositivesCount /
                    (truePositivesCount + falseNegativesCount)
                let precisionRate = (truePositivesCount + falsePositivesCount == 0) ? 0 : truePositivesCount /
                    (truePositivesCount + falsePositivesCount)
                perClassRecallRates.append(recallRate)
                perClassPrecisionRates.append(precisionRate)
            }

            let macroAverageRecallRate = perClassRecallRates.isEmpty ? 0 : perClassRecallRates
                .reduce(0, +) / Double(perClassRecallRates.count)
            let macroAveragePrecisionRate = perClassPrecisionRates.isEmpty ? 0 : perClassPrecisionRates
                .reduce(0, +) / Double(perClassPrecisionRates.count)

            print("    📊 検証データ マクロ平均再現率: \(String(format: "%.2f", macroAverageRecallRate * 100))%")
            print("    🎯 検証データ マクロ平均適合率: \(String(format: "%.2f", macroAveragePrecisionRate * 100))%")

            // .mlmodel のメタデータに含める shortDescription を動的に生成
            var modelMetadataShortDescription = String(
                format: "訓練正解率: %.1f%%, 検証正解率: %.1f%%",
                trainingDataAccuracyPercentage,
                validationDataAccuracyPercentage
            )
            if !labelsFromConfusion.isEmpty, macroAverageRecallRate > 0 || macroAveragePrecisionRate > 0 {
                modelMetadataShortDescription += String(
                    format: ", マクロ平均再現率: %.1f%%, マクロ平均適合率: %.1f%% (対象: %dクラス)",
                    macroAverageRecallRate * 100,
                    macroAveragePrecisionRate * 100,
                    labelsFromConfusion.count
                )
            }
            modelMetadataShortDescription += String(format: ", 総サンプル数: %d (自動分割)", totalImageSamples)

            let metadata = MLModelMetadata(
                author: author,
                shortDescription: modelMetadataShortDescription,
                version: version
            )

            let outputModelURL = finalOutputDir
                .appendingPathComponent("\(modelName)_\(classificationMethod)_\(version).mlmodel")

            print("  💾 [\(modelName)_\(classificationMethod)_\(version).mlmodel] を保存中: \(outputModelURL.path)")
            try model.write(to: outputModelURL, metadata: metadata)
            print("  ✅ [\(modelName)_\(classificationMethod)_\(version).mlmodel] は正常に保存されました。")

            return MultiClassTrainingResult(
                modelName: modelName,
                trainingDataAccuracy: trainingDataAccuracyPercentage,
                validationDataAccuracy: validationDataAccuracyPercentage,
                trainingDataErrorRate: trainingEvaluation.classificationError,
                validationDataErrorRate: validationEvaluation.classificationError,
                trainingTimeInSeconds: duration,
                modelOutputPath: outputModelURL.path,
                trainingDataPath: trainingDataParentDir.path,
                classLabels: classLabelsFromFileSystem,
                maxIterations: maxIterations,
                macroAverageRecall: macroAverageRecallRate,
                macroAveragePrecision: macroAveragePrecisionRate,
                detectedClassLabelsList: labelsFromConfusion
            )

        } catch let error as CreateML.MLCreateError {
            print("  ❌ モデル [\(modelName)] のトレーニングまたは保存エラー 。CreateMLエラー: \(error.localizedDescription)")
            return nil
        } catch {
            print("  ❌ トレーニングプロセス中に予期しないエラーが発生しました 。 \(error.localizedDescription)")
            if let nsError = error as NSError? {
                print("    詳細なエラー情報: \(nsError.userInfo)")
            }
            return nil
        }
    }
}

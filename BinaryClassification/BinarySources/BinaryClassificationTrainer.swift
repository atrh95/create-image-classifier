import CoreML
import CreateML
import CICConfusionMatrix
import CICInterface
import Foundation

public class BinaryClassificationTrainer: ScreeningTrainerProtocol {
    public typealias TrainingResultType = BinaryTrainingResult

    private let resourcesDirectoryPathOverride: String?
    private let outputDirectoryPathOverride: String?

    public var outputDirPath: String {
        if let overridePath = outputDirectoryPathOverride {
            return overridePath
        }
        var dir = URL(fileURLWithPath: #filePath)
        dir.deleteLastPathComponent()
        dir.deleteLastPathComponent()
        return dir.appendingPathComponent("OutputModels").path
    }

    public var classificationMethod: String { "Binary" }

    public var resourcesDirectoryPath: String {
        if let overridePath = resourcesDirectoryPathOverride {
            return overridePath
        }
        var dir = URL(fileURLWithPath: #filePath)
        dir.deleteLastPathComponent()
        dir.deleteLastPathComponent()
        return dir.appendingPathComponent("Resources").path
    }

    public init(resourcesDirectoryPathOverride: String? = nil, outputDirectoryPathOverride: String? = nil) {
        self.resourcesDirectoryPathOverride = resourcesDirectoryPathOverride
        self.outputDirectoryPathOverride = outputDirectoryPathOverride
    }

    private func createOutputDirectory(modelName: String, version: String) throws -> URL {
        let baseDirURL = URL(fileURLWithPath: outputDirPath)
            .appendingPathComponent(modelName)
            .appendingPathComponent(version)

        let fileManager = FileManager.default
        var resultNumber = 1

        // 既存のディレクトリを確認して次の番号を決定
        do {
            let contents = try fileManager.contentsOfDirectory(at: baseDirURL, includingPropertiesForKeys: nil)
            let existingNumbers = contents.compactMap { url -> Int? in
                let dirName = url.lastPathComponent
                guard dirName.hasPrefix("\(classificationMethod)_Result_") else { return nil }
                let numberStr = dirName.replacingOccurrences(of: "\(classificationMethod)_Result_", with: "")
                return Int(numberStr)
            }

            if let maxNumber = existingNumbers.max() {
                resultNumber = maxNumber + 1
            }
        } catch {
            // ディレクトリが存在しない場合は1から開始
            resultNumber = 1
        }

        let outputDirURL = baseDirURL.appendingPathComponent("\(classificationMethod)_Result_\(resultNumber)")

        try fileManager.createDirectory(
            at: outputDirURL,
            withIntermediateDirectories: true,
            attributes: nil
        )

        return outputDirURL
    }

    public func train(
        author: String,
        modelName: String,
        version: String,
        modelParameters: CreateML.MLImageClassifier.ModelParameters,
        scenePrintRevision: Int?
    ) async -> BinaryTrainingResult? {
        let resourcesPath = resourcesDirectoryPath
        let resourcesDirURL = URL(fileURLWithPath: resourcesPath)

        print("📁 リソースディレクトリ: \(resourcesPath)")

        // 出力ディレクトリ設定
        let outputDirectoryURL: URL
        do {
            outputDirectoryURL = try createOutputDirectory(
                modelName: modelName,
                version: version
            )
            print("📁 出力ディレクトリ作成成功: \(outputDirectoryURL.path)")
        } catch {
            print("❌ エラー: 出力ディレクトリ設定に失敗 \(error.localizedDescription)")
            return nil
        }

        print("🚀 Binaryトレーニング開始 (バージョン: \(version))...")

        let classLabelDirURLs: [URL]
        do {
            classLabelDirURLs = try FileManager.default.contentsOfDirectory(
                at: resourcesDirURL,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: .skipsHiddenFiles
            ).filter { url in
                var isDirectory: ObjCBool = false
                FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory)
                return isDirectory.boolValue && !url.lastPathComponent.hasPrefix(".")
            }
            print("📁 検出されたクラスラベルディレクトリ: \(classLabelDirURLs.map(\.lastPathComponent).joined(separator: ", "))")
        } catch {
            print("🛑 エラー: リソースディレクトリ内ラベルディレクトリ取得失敗: \(error.localizedDescription)")
            return nil
        }

        guard classLabelDirURLs.count == 2 else {
            print("🛑 エラー: Binary分類には2つのクラスラベルディレクトリが必要です。現在 \(classLabelDirURLs.count)個。処理中止。")
            return nil
        }

        let trainingDataParentDirURL = classLabelDirURLs[0].deletingLastPathComponent()
        print("📁 トレーニングデータ親ディレクトリ: \(trainingDataParentDirURL.path)")

        let trainingDataSource = MLImageClassifier.DataSource.labeledDirectories(at: trainingDataParentDirURL)
        print("📊 トレーニングデータソース作成完了")

        do {
            print("🔄 モデルトレーニング開始...")
            let trainingStartTime = Date()
            let imageClassifier = try MLImageClassifier(trainingData: trainingDataSource, parameters: modelParameters)
            let trainingEndTime = Date()
            let trainingDurationSeconds = trainingEndTime.timeIntervalSince(trainingStartTime)
            print("✅ モデルトレーニング完了 (所要時間: \(String(format: "%.1f", trainingDurationSeconds))秒)")

            let trainingMetrics = imageClassifier.trainingMetrics
            let validationMetrics = imageClassifier.validationMetrics

            // 混同行列の計算をCSBinaryConfusionMatrixに委任
            let confusionMatrix = CSBinaryConfusionMatrix(
                dataTable: validationMetrics.confusion,
                predictedColumn: "Predicted",
                actualColumn: "True Label"
            )

            // トレーニング完了後のパフォーマンス指標を表示
            print("\n📊 トレーニング結果サマリー")
            print(String(
                format: "  訓練正解率: %.1f%%",
                (1.0 - trainingMetrics.classificationError) * 100.0
            ))

            if let confusionMatrix {
                print(String(
                    format: "  検証正解率: %.1f%%",
                    confusionMatrix.accuracy * 100.0
                ))
                // 混同行列の表示
                print(confusionMatrix.getMatrixGraph())
            } else {
                print("⚠️ 警告: 検証データが不十分なため、混同行列の計算をスキップしました")
            }

            // データ拡張の説明
            let augmentationFinalDescription = if !modelParameters.augmentationOptions.isEmpty {
                String(describing: modelParameters.augmentationOptions)
            } else {
                "なし"
            }

            // 特徴抽出器の説明
            let featureExtractorDescription = String(describing: modelParameters.featureExtractor)
            let featureExtractorDesc: String = if let revision = scenePrintRevision {
                "\(featureExtractorDescription)(revision: \(revision))"
            } else {
                featureExtractorDescription
            }

            // モデルのメタデータを作成
            let modelMetadata = MLModelMetadata(
                author: author,
                shortDescription: """
                クラス: \(classLabelDirURLs.map(\.lastPathComponent).joined(separator: ", "))
                訓練正解率: \(String(format: "%.1f%%", (1.0 - trainingMetrics.classificationError) * 100.0))
                検証正解率: \(String(format: "%.1f%%", (1.0 - validationMetrics.classificationError) * 100.0))
                \(confusionMatrix.map { matrix in
                    "性能指標: [再現率: \(String(format: "%.1f%%", matrix.recall * 100.0)), " +
                        "適合率: \(String(format: "%.1f%%", matrix.precision * 100.0)), " +
                        "F1スコア: \(String(format: "%.1f%%", matrix.f1Score * 100.0))]"
                } ?? "")
                データ拡張: \(augmentationFinalDescription)
                特徴抽出器: \(featureExtractorDesc)
                """,
                version: version
            )

            let modelFileName = "\(modelName)_\(classificationMethod)_\(version).mlmodel"
            let modelFilePath = outputDirectoryURL.appendingPathComponent(modelFileName).path

            print("💾 モデルファイル保存中: \(modelFilePath)")
            try imageClassifier.write(to: URL(fileURLWithPath: modelFilePath), metadata: modelMetadata)
            print("✅ モデルファイル保存完了")

            return BinaryTrainingResult(
                modelName: modelName,
                trainingDataAccuracyPercentage: (1.0 - trainingMetrics.classificationError) * 100.0,
                validationDataAccuracyPercentage: (1.0 - validationMetrics.classificationError) * 100.0,
                trainingDataMisclassificationRate: trainingMetrics.classificationError,
                validationDataMisclassificationRate: validationMetrics.classificationError,
                trainingDurationInSeconds: trainingDurationSeconds,
                trainedModelFilePath: modelFilePath,
                sourceTrainingDataDirectoryPath: trainingDataParentDirURL.path,
                detectedClassLabelsList: classLabelDirURLs.map(\.lastPathComponent),
                maxIterations: modelParameters.maxIterations,
                dataAugmentationDescription: augmentationFinalDescription,
                featureExtractorDescription: featureExtractorDesc,
                confusionMatrix: confusionMatrix
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
}

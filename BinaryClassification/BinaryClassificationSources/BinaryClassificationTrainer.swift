import CoreML
import CreateML
import CSInterface
import Foundation

// MARK: - 画像二値分類トレーニング実行クラス

public class BinaryClassificationTrainer: ScreeningTrainerProtocol {
    public typealias TrainingResultType = BinaryTrainingResult

    // モデル名
    public var modelName: String { "ScaryCatScreeningML_Binary" }
    // カスタムモデル出力先ディレクトリパス
    public var customOutputDirPath: String { "BinaryClassification/OutputModels" }
    // 実行時出力名プレフィックス
    public var outputRunNamePrefix: String { "Binary" }

    // トレーニングリソース格納ディレクトリパス
    public var resourcesDirectoryPath: String {
        var dir = URL(fileURLWithPath: #filePath)
        dir.deleteLastPathComponent() // BinaryClassificationSources 削除
        dir.deleteLastPathComponent() // BinaryClassification 削除
        return dir.appendingPathComponent("Resources").path
    }

    public init() {}

    /// トレーニング処理
    /// - Parameters:
    ///   - author: モデル作成者名
    ///   - version: モデルバージョン
    ///   - maxIterations: トレーニング最大反復回数
    /// - Returns: トレーニング結果。失敗時は nil。
    public func train(
        author: String,
        version: String,
        maxIterations: Int
    ) async -> BinaryTrainingResult? {
        let resourcesPath = resourcesDirectoryPath
        let resourcesDirURL = URL(fileURLWithPath: resourcesPath)

        // 出力ディレクトリ設定
        let outputDirectoryURL: URL
        do {
            outputDirectoryURL = try setupVersionedRunOutputDirectory(
                version: version,
                trainerFilePath: #filePath
            )
        } catch {
            print("❌ エラー: 出力ディレクトリ設定に失敗 \(error.localizedDescription)")
            return nil
        }

        print("🚀 \(modelName) トレーニング開始...")

        // 主要トレーニング処理実行
        return await executeTrainingCore(
            trainingDataParentDirURL: resourcesDirURL,
            outputDirURL: outputDirectoryURL,
            author: author,
            version: version,
            maxIterations: maxIterations
        )
    }


    /// 主要なトレーニング処理
    private func executeTrainingCore(
        trainingDataParentDirURL: URL,
        outputDirURL: URL,
        author: String,
        version: String,
        maxIterations: Int
    ) async -> BinaryTrainingResult? {
        // トレーニングデータ親ディレクトリ存在確認
        guard FileManager.default.fileExists(atPath: trainingDataParentDirURL.path) else {
            print("❌ エラー: \(modelName) トレーニングデータ親ディレクトリが見つかりません: \(trainingDataParentDirURL.path)")
            return nil
        }

        // デバッグ用: トレーニングデータ親ディレクトリ内容表示試行
        do {
            _ = try FileManager.default.contentsOfDirectory(atPath: trainingDataParentDirURL.path)
        } catch {
            print("⚠️ 警告: トレーニングデータ親ディレクトリ内容表示失敗: \(error.localizedDescription)")
            // 処理続行
        }

        // サブディレクトリをクラスラベルとしてデータソース作成
        let trainingDataSource = MLImageClassifier.DataSource.labeledDirectories(at: trainingDataParentDirURL)

        do {
            let trainingStartTime = Date()

            // モデルパラメータ
            var modelParameters = MLImageClassifier.ModelParameters()
            modelParameters.featureExtractor = .scenePrint(revision: 1) // 特徴抽出器
            modelParameters.maxIterations = maxIterations
            modelParameters.validation = .split(strategy: .automatic) // 検証データ自動分割

            print("⏳ \(modelName) モデルトレーニング実行中 (最大反復: \(maxIterations)回)... ")
            let imageClassifier = try MLImageClassifier(trainingData: trainingDataSource, parameters: modelParameters)
            print("✅ \(modelName) モデルトレーニング完了")

            let trainingEndTime = Date()
            let trainingDurationSeconds = trainingEndTime.timeIntervalSince(trainingStartTime)

            print("🎉 \(modelName) トレーニング成功 (所要時間: \(String(format: "%.2f", trainingDurationSeconds))秒)")

            // 評価指標
            let trainingMetrics = imageClassifier.trainingMetrics
            let validationMetrics = imageClassifier.validationMetrics

            let trainingAccuracyPercentage = (1.0 - trainingMetrics.classificationError) * 100.0
            print("  📊 トレーニングデータ正解率: \(String(format: "%.2f", trainingAccuracyPercentage))%")

            let validationAccuracyPercentage = (1.0 - validationMetrics.classificationError) * 100.0
            print("  📈 検証データ正解率: \(String(format: "%.2f", validationAccuracyPercentage))%")

            var recallRate = 0.0
            var precisionRate = 0.0

            let confusionMatrix = validationMetrics.confusion
            print("デバッグ: 混同行列の内容: \(confusionMatrix.description)")
            print("デバッグ: 混同行列の列名: \(confusionMatrix.columnNames)")

            // MLDataTableの行構成: actualLabel | predictedLabel | count
            var labelSet = Set<String>()
            var rowCount = 0
            for row in confusionMatrix.rows {
                rowCount += 1
                // print("デバッグ: 混同行列の処理中の行: \(row)")
                if let actual = row["True Label"]?.stringValue {
                    labelSet.insert(actual)
                }
                if let predicted = row["Predicted"]?.stringValue {
                    labelSet.insert(predicted)
                }
            }
            print("デバッグ: 混同行列から処理された総行数: \(rowCount)")
            print("デバッグ: 混同行列から抽出されたラベルセット: \(labelSet)")
            let classLabelsFromConfusion = Array(labelSet).sorted()

            // 二値分類の場合、再現率と適合率を計算
            if classLabelsFromConfusion.count == 2 {
                // classLabelsFromConfusion はソート済み想定 (例: ["Negative", "Positive"])
                // 2番目のラベルを陽性クラスとする
                let negativeLabel = classLabelsFromConfusion[0]
                let positiveLabel = classLabelsFromConfusion[1]

                var truePositives = 0
                var falsePositives = 0
                var falseNegatives = 0

                for row in confusionMatrix.rows {
                    guard
                        let actual = row["True Label"]?.stringValue,
                        let predicted = row["Predicted"]?.stringValue,
                        let cnt = row["count"]?.intValue
                    else { continue }

                    if actual == positiveLabel, predicted == positiveLabel {
                        truePositives += cnt
                    } else if actual == negativeLabel, predicted == positiveLabel {
                        falsePositives += cnt
                    } else if actual == positiveLabel, predicted == negativeLabel {
                        falseNegatives += cnt
                    }
                }

                if (truePositives + falseNegatives) > 0 {
                    recallRate = Double(truePositives) / Double(truePositives + falseNegatives)
                }
                if (truePositives + falsePositives) > 0 {
                    precisionRate = Double(truePositives) / Double(truePositives + falsePositives)
                }
                print("    🔍 検証データ 再現率 (陽性クラス: \(positiveLabel)): \(String(format: "%.2f", recallRate * 100))%")
                print("    🎯 検証データ 適合率 (陽性クラス: \(positiveLabel)): \(String(format: "%.2f", precisionRate * 100))%")
            } else {
                print("    ⚠️ 再現率・適合率は二値分類の場合のみ計算 (現在クラス数: \(classLabelsFromConfusion.count))")
            }

            // 各クラスの画像枚数とクラス名リスト取得
            var imageCountsPerClass: [String: Int] = [:]
            var classNamesFromDataDirs: [String] = []

            let classLabelDirURLs = (
                try? FileManager.default.contentsOfDirectory(
                    at: trainingDataParentDirURL,
                    includingPropertiesForKeys: [.isDirectoryKey],
                    options: .skipsHiddenFiles
                )
                .filter { url in
                    var isDir: ObjCBool = false
                    FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir)
                    return isDir.boolValue && !url.lastPathComponent.hasPrefix(".")
                }
                .sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) // 名前でソートし一貫性を保持
            ) ?? []

            for labelDirURL in classLabelDirURLs {
                let className = labelDirURL.lastPathComponent
                classNamesFromDataDirs.append(className)
                if let files = try? FileManager.default.contentsOfDirectory(
                    at: labelDirURL,
                    includingPropertiesForKeys: [.isRegularFileKey],
                    options: .skipsHiddenFiles
                ) {
                    imageCountsPerClass[className] = files.filter { !$0.hasDirectoryPath }.count
                } else {
                    imageCountsPerClass[className] = 0
                }
            }

            // .mlmodel メタデータ用 shortDescription 生成
            var modelMetadataShortDescription = String(
                format: "訓練正解率: %.1f%%, 検証正解率: %.1f%%",
                trainingAccuracyPercentage,
                validationAccuracyPercentage
            )

            if classLabelsFromConfusion.count == 2 {
                // 2番目のラベルを陽性クラスとして使用
                let positiveLabelForDesc = classLabelsFromConfusion[1]
                modelMetadataShortDescription += String(format: "\n陽性クラス: %@, 再現率: %.1f%%, 適合率: %.1f%%",
                                                     positiveLabelForDesc,
                                                     recallRate * 100,
                                                     precisionRate * 100)
            } else if !classLabelsFromConfusion.isEmpty {
                 modelMetadataShortDescription += "\n(詳細な分類指標は二値分類のみ)"
            }

            // クラス構成情報追加
            if !classNamesFromDataDirs.isEmpty {
                let classCountsStrings = classNamesFromDataDirs.map { className in
                    let count = imageCountsPerClass[className] ?? 0
                    return "\(className): \(count)枚"
                }
                modelMetadataShortDescription += "\nクラス構成: " + classCountsStrings.joined(separator: "; ")
            } else {
                modelMetadataShortDescription += "\nクラス構成情報なし"
            }
            
            modelMetadataShortDescription += "\n(検証: 自動分割)"

            let modelMetadata = MLModelMetadata(
                author: author,
                shortDescription: modelMetadataShortDescription,
                version: version
            )

            let outputModelFileURL = outputDirURL.appendingPathComponent("\(modelName)_\(version).mlmodel")

            print("💾 \(modelName) (v\(version)) 保存中: \(outputModelFileURL.path)")
            try imageClassifier.write(to: outputModelFileURL, metadata: modelMetadata)
            print("✅ \(modelName) (v\(version)) 保存完了")

            // 結果レポート用にデータディレクトリ由来のクラスラベルリストを採用
            let detectedClassLabels = classNamesFromDataDirs

            // トレーニング結果返却
            return BinaryTrainingResult(
                modelName: modelName,
                trainingDataAccuracyPercentage: trainingAccuracyPercentage,
                validationDataAccuracyPercentage: validationAccuracyPercentage,
                trainingDataMisclassificationRate: trainingMetrics.classificationError,
                validationDataMisclassificationRate: validationMetrics.classificationError,
                trainingDurationInSeconds: trainingDurationSeconds,
                trainedModelFilePath: outputModelFileURL.path,
                sourceTrainingDataDirectoryPath: trainingDataParentDirURL.path,
                detectedClassLabelsList: detectedClassLabels,
                maxIterations: maxIterations
            )

        } catch let createMLError as CreateML.MLCreateError { // CreateML固有エラー
            switch createMLError {
                case .io:
                    print("❌ \(modelName) 保存エラー (I/O): \(createMLError.localizedDescription)")
                default:
                    print("❌ \(modelName) トレーニングエラー (CreateML): \(createMLError.localizedDescription)")
                    print("  詳細情報: \(createMLError)")
            }
            return nil
        } catch { // その他エラー
            print("❌ \(modelName) トレーニング/保存中に予期しないエラー: \(error.localizedDescription)")
            return nil
        }
    }
}

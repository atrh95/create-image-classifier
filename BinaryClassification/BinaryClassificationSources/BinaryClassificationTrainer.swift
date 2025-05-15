import CoreML
import CreateML
import CSInterface
import Foundation

// MARK: - 画像二値分類トレーニング実行クラス

public class BinaryClassificationTrainer: ScreeningTrainerProtocol {
    public typealias TrainingResultType = BinaryTrainingResult

    // 生成するモデル名
    public var modelName: String { "ScaryCatScreeningML_Binary" }
    // カスタムモデルの出力先ディレクトリパス
    public var customOutputDirPath: String { "BinaryClassification/OutputModels" }
    // 実行時の出力名プレフィックス
    public var outputRunNamePrefix: String { "Binary" }

    // トレーニング用リソース (画像データなど) が格納されているディレクトリのパス
    public var resourcesDirectoryPath: String {
        var dir = URL(fileURLWithPath: #filePath) // このファイルのパスを取得
        dir.deleteLastPathComponent() // "BinaryClassificationSources" を削除
        dir.deleteLastPathComponent() // "BinaryClassification" を削除
        return dir.appendingPathComponent("Resources").path // "Resources" フォルダのパスを返す
    }

    public init() {}

    /// トレーニング処理を実行します。
    /// - Parameters:
    ///   - author: モデルの作成者名
    ///   - version: モデルのバージョン
    ///   - maxIterations: トレーニングの最大反復回数
    /// - Returns: トレーニング結果。失敗した場合は nil。
    public func train(
        author: String,
        version: String,
        maxIterations: Int
    ) async -> BinaryTrainingResult? {
        let resourcesPath = resourcesDirectoryPath
        let resourcesDirURL = URL(fileURLWithPath: resourcesPath) // リソースディレクトリのURL

        // --- 出力ディレクトリの設定 ---
        let outputDirectoryURL: URL
        do {
            // バージョン管理された実行結果出力ディレクトリを設定
            outputDirectoryURL = try setupVersionedRunOutputDirectory(
                version: version,
                trainerFilePath: #filePath // このファイルのパスを渡して、trainer名を取得
            )
        } catch {
            print("❌ エラー: 出力ディレクトリの設定に失敗しました \(error.localizedDescription)")
            return nil
        }
        // --- 出力ディレクトリ設定完了 ---

        print("🚀 \(modelName) のトレーニングを開始します...")

        // 主要なトレーニング処理を実行
        return await executeTrainingCore(
            trainingDataParentDirURL: resourcesDirURL, // トレーニングデータが格納されている親ディレクトリのURL
            outputDirURL: outputDirectoryURL, // モデルや結果の出力先ディレクトリURL
            author: author,
            version: version,
            maxIterations: maxIterations
        )
    }

    // MARK: - プライベート補助メソッド

    /// 主要なトレーニング処理を実行します。
    private func executeTrainingCore(
        trainingDataParentDirURL: URL,
        outputDirURL: URL,
        author: String,
        version: String,
        maxIterations: Int
    ) async -> BinaryTrainingResult? {
        // トレーニングデータの親ディレクトリが存在するか確認
        guard FileManager.default.fileExists(atPath: trainingDataParentDirURL.path) else {
            print("❌ エラー: \(modelName) のトレーニングデータ親ディレクトリが見つかりません 。 \(trainingDataParentDirURL.path)")
            return nil
        }

        // トレーニングデータの親ディレクトリの内容をリストしようと試みる（デバッグ用）
        do {
            _ = try FileManager.default.contentsOfDirectory(atPath: trainingDataParentDirURL.path)
        } catch {
            print("⚠️ 警告: トレーニングデータ親ディレクトリの内容をリストできませんでした 。 \(error.localizedDescription)")
            // ここでは処理を中断せず、続行する
        }

        // CreateML用のトレーニングデータソースを作成
        // trainingDataParentDirURL 内の各サブディレクトリがクラスラベルとなる
        let trainingDataSource = MLImageClassifier.DataSource.labeledDirectories(at: trainingDataParentDirURL)

        do {
            // --- トレーニングと評価 ---
            let trainingStartTime = Date()

            // モデルのパラメータ設定
            var modelParameters = MLImageClassifier.ModelParameters()
            modelParameters.featureExtractor = .scenePrint(revision: 1) // 特徴抽出器として ScenePrint を使用
            modelParameters.maxIterations = maxIterations // 最大反復回数
            modelParameters.validation = .split(strategy: .automatic) // 検証データの分割戦略 (自動)

            // モデルのトレーニングを実行
            print("⏳ \(modelName) のモデルトレーニングを実行中... (最大反復: \(maxIterations)回)")
            let imageClassifier = try MLImageClassifier(trainingData: trainingDataSource, parameters: modelParameters)
            print("✅ \(modelName) のモデルトレーニングが完了しました。")

            let trainingEndTime = Date()
            let trainingDurationSeconds = trainingEndTime.timeIntervalSince(trainingStartTime)

            print("🎉 \(modelName) のトレーニングに成功しました！ (所要時間: \(String(format: "%.2f", trainingDurationSeconds))秒)")

            // トレーニング結果の評価指標を取得
            let trainingMetrics = imageClassifier.trainingMetrics
            let validationMetrics = imageClassifier.validationMetrics

            let trainingAccuracyPercentage = (1.0 - trainingMetrics.classificationError) * 100.0
            let trainingAccuracyPercentageString = String(format: "%.2f", trainingAccuracyPercentage)
            print("  📊 トレーニングデータ正解率: \(trainingAccuracyPercentageString)%")

            let validationAccuracyPercentage = (1.0 - validationMetrics.classificationError) * 100.0
            let validationAccuracyPercentageString = String(format: "%.2f", validationAccuracyPercentage)
            print("  📈 検証データ正解率: \(validationAccuracyPercentageString)%")

            // 再現率 (Recall) と適合率 (Precision) の計算
            var recallRate = 0.0
            var precisionRate = 0.0

            let confusionMatrix = validationMetrics.confusion // 混同行列を取得

            // MLDataTable では各行が actualLabel | predictedLabel | count の 3 列構成
            var labelSet = Set<String>()
            for row in confusionMatrix.rows {
                if let actual = row["actualLabel"]?.stringValue {
                    labelSet.insert(actual)
                }
                if let predicted = row["predictedLabel"]?.stringValue {
                    labelSet.insert(predicted)
                }
            }
            let classLabelsFromConfusion = Array(labelSet).sorted()

            // 二値分類の場合のみ再現率と適合率を計算
            if classLabelsFromConfusion.count == 2 {
                // classLabelsFromConfusion はアルファベット順などでソートされている想定
                // 例: ["Negative", "Positive"] や ["Cat", "Dog"]
                // どちらのラベルを陽性 (Positive) とみなすかは、データの構成に依存
                // ここでは、便宜上、2番目のラベルを陽性クラスとする
                let negativeLabel = classLabelsFromConfusion[0]
                let positiveLabel = classLabelsFromConfusion[1]

                var truePositives = 0
                var falsePositives = 0
                var falseNegatives = 0

                for row in confusionMatrix.rows {
                    guard
                        let actual = row["actualLabel"]?.stringValue,
                        let predicted = row["predictedLabel"]?.stringValue,
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
                print("    ⚠️ 再現率・適合率は二値分類の場合のみ計算されます。(現在のクラス数: \(classLabelsFromConfusion.count))")
            }
            // --- トレーニングと評価 完了 ---

            // トレーニングに使用した総サンプル数を計算
            var totalImageSamples = 0
            let classLabelDirs = (
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
            ) ?? []

            for labelDirURL in classLabelDirs {
                if let files = try? FileManager.default.contentsOfDirectory(
                    at: labelDirURL,
                    includingPropertiesForKeys: [.isRegularFileKey],
                    options: .skipsHiddenFiles
                ) {
                    totalImageSamples += files.filter { !$0.hasDirectoryPath }.count
                }
            }

            // .mlmodel のメタデータに含める shortDescription を動的に生成
            var modelMetadataShortDescription = String(
                format: "訓練正解率: %.1f%%, 検証正解率: %.1f%%",
                trainingAccuracyPercentage,
                validationAccuracyPercentage
            )
            if classLabelsFromConfusion.count == 2 {
                let positiveLabelForDesc = classLabelsFromConfusion[1]
                var metricsSummary = ""
                if recallRate > 0 || precisionRate > 0 {
                    metricsSummary = String(
                        format: ", 再現率(%@): %.1f%%, 適合率(%@): %.1f%%",
                        positiveLabelForDesc, recallRate * 100,
                        positiveLabelForDesc, precisionRate * 100
                    )
                }
                modelMetadataShortDescription += metricsSummary
            } else {
                modelMetadataShortDescription += " (詳細指標対象外)"
            }
            modelMetadataShortDescription += String(format: ", 総サンプル数: %d (検証自動分割)", totalImageSamples)

            // モデルのメタデータを作成
            let modelMetadata = MLModelMetadata(
                author: author,
                shortDescription: modelMetadataShortDescription, // 動的に生成した説明文を使用
                version: version
            )

            // 学習済みモデルの出力先ファイルURLを決定
            let outputModelFileURL = outputDirURL.appendingPathComponent("\(modelName)_\(version).mlmodel")

            print("💾 \(modelName) (バージョン: \(version)) を保存中: \(outputModelFileURL.path)")
            try imageClassifier.write(to: outputModelFileURL, metadata: modelMetadata)
            print("✅ \(modelName) (バージョン: \(version)) は正常に保存されました。")

            // トレーニングに使用したディレクトリ名からクラスラベルを取得 (結果レポート用)
            let detectedClassLabels: [String]
            do {
                let directoryContents = try FileManager.default
                    .contentsOfDirectory(atPath: trainingDataParentDirURL.path)
                // 隠しファイルを除外し、ディレクトリのみをフィルタリング & ソート
                detectedClassLabels = directoryContents.filter { itemName in
                    var isDirectory: ObjCBool = false
                    let fullItemPath = trainingDataParentDirURL.appendingPathComponent(itemName).path
                    // ドットで始まらない、かつ、ディレクトリであるものを抽出
                    return !itemName.hasPrefix(".") &&
                        FileManager.default.fileExists(atPath: fullItemPath, isDirectory: &isDirectory) &&
                        isDirectory.boolValue
                }.sorted() // アルファベット順にソート
            } catch {
                print("⚠️ クラスラベルの取得に失敗しました (ディレクトリ: \(trainingDataParentDirURL.path)) 。 \(error.localizedDescription)")
                detectedClassLabels = [] // 失敗した場合は空の配列
            }

            // トレーニング結果をまとめる
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

        } catch let createMLError as CreateML.MLCreateError { // CreateML固有のエラー処理
            switch createMLError {
                case .io:
                    print("❌ モデル \(modelName) の保存エラー 。 I/Oエラー: \(createMLError.localizedDescription)")
                // 他のCreateMLエラーケースも必要に応じて追加
                default:
                    print("❌ モデル \(self.modelName) のトレーニングエラー 。 不明なCreateMLエラー: \(createMLError.localizedDescription)")
                    print("  詳細なCreateMLエラー情報: \(createMLError)")
            }
            return nil
        } catch { // その他の予期しないエラー
            print("❌ \(modelName) のトレーニングまたは保存中に予期しないエラーが発生しました 。 \(error.localizedDescription)")
            return nil
        }
    }
}

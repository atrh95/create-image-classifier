import Foundation
import SCSInterface 

/// 特定のラベルに対する評価
public struct LabelMetrics {
    public let precision: Double
    public let recall: Double
    public let f1Score: Double
    public let support: Int
}

/// 画像分類モデルのトレーニング結果（ログ記録用）を格納する構造体
public struct MultiClassTrainingResult: TrainingResultData {
    /// トレーニングデータでの正解率 (0.0 ~ 100.0)
    public let trainingAccuracy: Double
    /// 検証データでの正解率 (0.0 ~ 100.0)
    public let validationAccuracy: Double
    /// トレーニングデータでのエラー率 (0.0 ~ 1.0)
    public let trainingError: Double
    /// 検証データでのエラー率 (0.0 ~ 1.0)
    public let validationError: Double
    /// トレーニングにかかった時間（秒）
    public let trainingDuration: TimeInterval
    /// 生成されたモデルファイルの出力パス
    public let modelOutputPath: String
    /// トレーニングに使用されたデータのパス
    public let trainingDataPath: String
    /// 検出されたクラスラベルのリスト
    public let classLabels: [String]
    /// 各ラベルごとの詳細なメトリクス (オプション)
    public let perLabelMetrics: [String: LabelMetrics]?

    public func saveLog(trainer: any ScreeningTrainerProtocol, modelAuthor: String, modelDescription: String, modelVersion: String) {
        let modelName = trainer.modelName

        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd HH:mm:ss Z"
        let generatedDateString = dateFormatter.string(from: Date())

        let classLabelsString = classLabels.isEmpty ? "不明" : classLabels.joined(separator: ", ")

        let trainingAccStr = String(format: "%.2f", trainingAccuracy)
        let validationAccStr = String(format: "%.2f", validationAccuracy)
        let trainingErrStr = String(format: "%.2f", trainingError * 100)
        let validationErrStr = String(format: "%.2f", validationError * 100)
        let durationStr = String(format: "%.2f", trainingDuration)

        var markdownText = """
        # モデルトレーニング情報: \(modelName)

        ## モデル詳細
        モデル名           : \(modelName)
        保存先モデルパス   : \(modelOutputPath)
        ファイル生成日時   : \(generatedDateString)

        ## トレーニング設定
        訓練データパス     : \(trainingDataPath)
        使用されたクラスラベル : \(classLabelsString)

        ## パフォーマンス指標 (全体)
        トレーニング所要時間: \(durationStr) 秒
        トレーニングエラー率 (学習時) : \(trainingErrStr)%
        訓練データ正解率 (学習時) : \(trainingAccStr)%
        検証データ正解率 (学習時自動検証) : \(validationAccStr)%
        検証誤分類率 (学習時自動検証) : \(validationErrStr)%
        """

        if let perLabelMetrics = perLabelMetrics, !perLabelMetrics.isEmpty {
            markdownText += "\n\n        ## パフォーマンス指標 (各ラベル別 - 手動検証セット)"
            markdownText += "\n        | ラベル名 | 適合率 (Precision) | 再現率 (Recall) | F1スコア | サポート数 (Support) |\n"
            markdownText += "        |:-------|:-----------------:|:--------------:|:--------:|:-----------------:|\n"

            // ラベル名でソートして一貫した順序で表示
            let sortedLabels = perLabelMetrics.keys.sorted()
            for labelName in sortedLabels {
                if let metrics = perLabelMetrics[labelName] {
                    let precisionStr = String(format: "%.3f", metrics.precision)
                    let recallStr = String(format: "%.3f", metrics.recall)
                    let f1ScoreStr = String(format: "%.3f", metrics.f1Score)
                    markdownText +=
                        "        | \(labelName) | \(precisionStr) | \(recallStr) | \(f1ScoreStr) | \(metrics.support) |\n"
                }
            }
        }

        markdownText += """

        ## モデルメタデータ
        作成者            : \(modelAuthor)
        説明              : \(modelDescription)
        バージョン          : \(modelVersion)
        """

        let outputDir = URL(fileURLWithPath: modelOutputPath).deletingLastPathComponent()
        let textFileName = "\(modelName).md"
        let textFilePath = outputDir.appendingPathComponent(textFileName).path

        do {
            try markdownText.write(toFile: textFilePath, atomically: true, encoding: String.Encoding.utf8)
            print("✅ [\(modelName)] モデル情報をMarkdownファイルに保存しました: \(textFilePath)")
        } catch {
            print("❌ [\(modelName)] Markdownファイルの書き込みに失敗しました: \(error.localizedDescription)")
            print("--- [\(modelName)] モデル情報 (Markdown) ---:")
            print(markdownText)
            print("--- ここまで --- ")
        }
    }
}

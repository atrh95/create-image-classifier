import Foundation
import SCSInterface // For TrainingResultData and ScreeningTrainerProtocol

// Note: Create ML's MLImageClassifier.Metrics might not provide
// precision/recall/F1 for multi-label in the same way as multi-class.
// This struct mirrors TrainingResultLogModel for now and can be adjusted.

/// 画像分類モデルのトレーニング結果（マルチラベル用）を格納する構造体
public struct MultiLabelTrainingResult: TrainingResultData {
    /// トレーニングデータでの正解率 (0.0 ~ 100.0)
    public let trainingAccuracy: Double
    /// 検証データでの正解率 (0.0 ~ 100.0)
    public let validationAccuracy: Double // This might be less straightforward for multi-label
    /// トレーニングデータでのエラー率 (0.0 ~ 1.0)
    public let trainingError: Double
    /// 検証データでのエラー率 (0.0 ~ 1.0)
    public let validationError: Double // This might be less straightforward for multi-label
    /// トレーニングにかかった時間（秒）
    public let trainingDuration: TimeInterval
    /// 生成されたモデルファイルの出力パス
    public let modelOutputPath: String
    /// トレーニングに使用されたデータのパス (JSON manifest path)
    public let trainingDataPath: String
    /// 検出された全ユニーククラスラベルのリスト
    public let classLabels: [String]
    // Multi-label might have different ways of reporting per-label metrics, if at all directly.
    // For now, we'll omit perLabelMetrics or keep it nil.

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

        let markdownText = """
        # モデルトレーニング情報: \(modelName) (Multi-Label)

        ## モデル詳細
        モデル名           : \(modelName)
        保存先モデルパス   : \(modelOutputPath)
        ファイル生成日時   : \(generatedDateString)

        ## トレーニング設定
        訓練データ Manifest : \(trainingDataPath)
        検出された全ラベル : \(classLabelsString)

        ## パフォーマンス指標 (全体)
        トレーニング所要時間: \(durationStr) 秒
        トレーニングエラー率 (学習時) : \(trainingErrStr)%
        訓練データ正解率 (学習時) : \(trainingAccStr)%
        検証データ正解率 (学習時自動検証) : \(validationAccStr)% (注: マルチラベルの正解率は解釈に注意)
        検証誤分類率 (学習時自動検証) : \(validationErrStr)% (注: マルチラベルの誤分類率は解釈に注意)

        ## モデルメタデータ
        作成者            : \(modelAuthor)
        説明              : \(modelDescription)
        バージョン          : \(modelVersion)
        """
        // Note: Per-label metrics might need a different representation for multi-label.
        // This basic log doesn't include them yet.

        let outputDir = URL(fileURLWithPath: modelOutputPath).deletingLastPathComponent()
        let textFileName = "\(modelName)_\(modelVersion).md"
        let textFilePath = outputDir.appendingPathComponent(textFileName).path

        do {
            try markdownText.write(toFile: textFilePath, atomically: true, encoding: String.Encoding.utf8)
            print("✅ [\(modelName) - Multi-Label] モデル情報をMarkdownファイルに保存しました: \(textFilePath)")
        } catch {
            print("❌ [\(modelName) - Multi-Label] Markdownファイルの書き込みに失敗しました: \(error.localizedDescription)")
            print("--- [\(modelName) - Multi-Label] モデル情報 (Markdown) ---:")
            print(markdownText)
            print("--- ここまで --- ")
        }
    }
} 
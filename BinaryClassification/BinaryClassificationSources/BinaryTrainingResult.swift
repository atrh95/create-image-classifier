import Foundation
import SCSInterface // For TrainingResultData and ScreeningTrainerProtocol

/// 画像分類モデルのトレーニング結果を格納する構造体
public struct BinaryTrainingResult: TrainingResultProtocol {
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

    public func saveLog(trainer: any ScreeningTrainerProtocol, modelAuthor: String, modelDescription: String, modelVersion: String) {
        // ファイル生成日時フォーマッタ
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd HH:mm:ss Z"
        let generatedDateString = dateFormatter.string(from: Date())

        // resultからクラスラベルを取得 (trainer.classLabelsから変更)
        let classLabelsString = classLabels.isEmpty ? "不明" : classLabels.joined(separator: ", ")

        // 文字列フォーマット
        let trainingAccStr = String(format: "%.2f", trainingAccuracy)
        let validationAccStr = String(format: "%.2f", validationAccuracy)
        let trainingErrStr = String(format: "%.2f", trainingError * 100) // %表示に変更
        let validationErrStr = String(format: "%.2f", validationError * 100) // %表示に変更
        let durationStr = String(format: "%.2f", trainingDuration)

        // Markdownの内容
        let infoText = """
        # モデルトレーニング情報

        ## モデル詳細
        モデル名           : \(trainer.modelName)
        保存先モデルパス   : \(modelOutputPath)
        ファイル生成日時   : \(generatedDateString)

        ## トレーニング設定
        訓練データ親ディレクトリ: \(trainingDataPath)
        使用されたクラスラベル : \(classLabelsString)

        ## パフォーマンス指標
        トレーニング所要時間: \(durationStr) 秒
        トレーニングエラー率 : \(trainingErrStr)%
        訓練データ正解率 : \(trainingAccStr)% ※モデルが学習に使用したデータでの正解率
        検証データ正解率 : \(validationAccStr)% ※モデルが学習に使用していない未知のデータでの正解率
        検証誤分類率       : \(validationErrStr)%

        ## モデルメタデータ
        作成者            : \(modelAuthor)
        説明              : \(modelDescription)
        バージョン          : \(modelVersion)
        """

        // Markdownファイルのパスを作成 (モデルファイルと同じディレクトリ、拡張子を.mdに変更)
        let outputDir = URL(fileURLWithPath: modelOutputPath).deletingLastPathComponent()
        let textFileName = "\(trainer.modelName).md" // モデル名から .md ファイル名を生成
        let textFilePath = outputDir.appendingPathComponent(textFileName).path

        // Markdownファイルに書き込み
        do {
            try infoText.write(toFile: textFilePath, atomically: true, encoding: .utf8)
            print("✅ モデル情報をMarkdownファイルに保存しました: \(textFilePath)")
        } catch {
            print("❌ Markdownファイルの書き込みに失敗しました: \(error.localizedDescription)")
            // 書き込み失敗時はコンソールに出力
            print("--- モデル情報 (Markdown) ---:")
            print(infoText)
            print("--- ここまで --- ")
        }
    }
}

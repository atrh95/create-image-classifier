import CSInterface
import Foundation

public struct MultiLabelTrainingResult: TrainingResultProtocol {
    public let modelName: String
    public let trainingDataAccuracy: Double
    public let validationDataAccuracy: Double
    public let trainingDataError: Double
    public let validationDataError: Double
    public let trainingDuration: TimeInterval
    public let modelOutputPath: String
    public let trainingDataPath: String
    public let classLabels: [String]

    public func saveLog(
        modelAuthor: String,
        modelDescription: String,
        modelVersion: String
    ) {
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd HH:mm:ss Z"
        let generatedDateString = dateFormatter.string(from: Date())

        let classLabelsString = classLabels.isEmpty ? "不明" : classLabels.joined(separator: ", ")

        let trainingAccStr = String(format: "%.2f", trainingDataAccuracy)
        let validationAccStr = String(format: "%.2f", validationDataAccuracy)
        let trainingErrStr = String(format: "%.2f", trainingDataError * 100)
        let validationErrStr = String(format: "%.2f", validationDataError * 100)
        let durationStr = String(format: "%.2f", trainingDuration)

        let markdownText = """
        # モデルトレーニング情報: \(modelName) (Multi-Label)

        ## モデル詳細
        モデル名           : \(modelName)
        ファイル生成日時   : \(generatedDateString)

        ## トレーニング設定
        検出された全ラベル : \(classLabelsString)

        ## パフォーマンス指標 (全体)
        トレーニング所要時間: \(durationStr) 秒
        トレーニング誤分類率 (学習時) : \(trainingErrStr)%
        訓練データ正解率 (学習時) : \(trainingAccStr)%
        検証データ正解率 (学習時自動検証) : \(validationAccStr)% (注: マルチラベルの正解率は解釈に注意)
        検証誤分類率 (学習時自動検証) : \(validationErrStr)% (注: マルチラベルの誤分類率は解釈に注意)

        ## モデルメタデータ
        作成者            : \(modelAuthor)
        説明              : \(modelDescription)
        バージョン          : \(modelVersion)
        """

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

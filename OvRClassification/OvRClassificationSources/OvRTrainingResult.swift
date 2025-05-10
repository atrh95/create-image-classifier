import CSInterface
import Foundation

public struct IndividualModelReport {
    public let modelName: String
    public let trainingAccuracy: Double
    public let validationAccuracy: Double
}

public struct OvRTrainingResult: TrainingResultProtocol {
    public let modelOutputPath: String
    public let trainingDataAccuracy: Double
    public let validationDataAccuracy: Double
    public let trainingDataErrorRate: Double
    public let validationDataErrorRate: Double
    public let trainingTimeInSeconds: TimeInterval
    public let trainingDataPath: String
    public let maxIterations: Int
    public let individualReports: [IndividualModelReport]

    public func saveLog(modelAuthor: String, modelDescription: String, modelVersion: String) {
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd HH:mm:ss Z"
        let generatedDateString = dateFormatter.string(from: Date())

        let reportFileName =
            "OvR_Model_Stat_\(modelVersion).md"
        let modelDir = URL(fileURLWithPath: modelOutputPath).standardizedFileURL.deletingLastPathComponent()
        let reportURL = modelDir.appendingPathComponent(reportFileName)

        var markdownText = """
        # モデルトレーニング情報

        ## モデル詳細
        ファイル生成日時   : \(generatedDateString)
        最大反復回数     : \(maxIterations)

        ## パフォーマンス指標 (全体平均)
        トレーニング所要時間              : \(String(format: "%.2f", trainingTimeInSeconds)) 秒
        トレーニング誤分類率 (学習時)     : \(String(format: "%.2f", trainingDataErrorRate * 100))%
        訓練データ正解率 (学習時)         : \(String(format: "%.2f", trainingDataAccuracy * 100))%
        検証データ正解率 (学習時自動検証) : \(String(format: "%.2f", validationDataAccuracy * 100))%
        検証誤分類率 (学習時自動検証)     : \(String(format: "%.2f", validationDataErrorRate * 100))%
        """

        if !individualReports.isEmpty {
            markdownText += """

            ## 個別モデルのパフォーマンス指標
            | モデル名                        | 訓練データ正解率 | 検証データ正解率 | 最大反復回数 |
            |---------------------------------|--------------------|--------------------|--------------|
            """
            let iterationsStr = "\(self.maxIterations)"
            for report in individualReports {
                let modelFileName = report.modelName
                let trainingAccStr = String(
                    format: "%.4f (%.2f%%)",
                    report.trainingAccuracy,
                    report.trainingAccuracy * 100
                )
                let validationAccStr = String(
                    format: "%.4f (%.2f%%)",
                    report.validationAccuracy,
                    report.validationAccuracy * 100
                )
                markdownText +=
                    "\n| \(modelFileName.padding(toLength: 30, withPad: " ", startingAt: 0)) | \(trainingAccStr.padding(toLength: 18, withPad: " ", startingAt: 0)) | \(validationAccStr.padding(toLength: 18, withPad: " ", startingAt: 0)) | \(iterationsStr.padding(toLength: 12, withPad: " ", startingAt: 0)) |"
            }
            markdownText += "\n"
        }

        markdownText += """

        ## モデルメタデータ
        作成者            : \(modelAuthor)
        説明              : \(modelDescription)
        バージョン        : \(modelVersion)
        """

        do {
            try markdownText.write(to: reportURL, atomically: true, encoding: .utf8)
            print("✅ OvRレポートを保存しました: \(reportURL.path)")
        } catch {
            print("❌ OvRレポートの保存エラー: \(error.localizedDescription) (Path: \(reportURL.path))")
        }
    }
}

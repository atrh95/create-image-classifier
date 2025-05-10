import CSInterface

public struct OvRTrainingResult: TrainingResultProtocol {
    public let modelOutputPath: String
    public let trainingDataAccuracy: Double
    public let validationDataAccuracy: Double
    public let trainingDataErrorRate: Double
    public let validationDataErrorRate: Double
    public let trainingTimeInSeconds: TimeInterval
    public let trainingDataPath: String

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

        ## パフォーマンス指標 (全体)
        トレーニング所要時間              : \(String(format: "%.2f", trainingTimeInSeconds)) 秒
        トレーニング誤分類率 (学習時)     : \(String(format: "%.2f", trainingDataErrorRate * 100))%
        訓練データ正解率 (学習時)         : \(String(format: "%.2f", trainingDataAccuracy * 100))%
        検証データ正解率 (学習時自動検証) : \(String(format: "%.2f", validationDataAccuracy * 100))%
        検証誤分類率 (学習時自動検証)     : \(String(format: "%.2f", validationDataErrorRate * 100))%

        ## モデルメタデータ
        作成者            : \(modelAuthor)
        説明              : \(modelDescription)
        バージョン        : \(modelVersion)
        """

        do {
            try markdownText.write(to: reportURL, atomically: true, encoding: .utf8)
            print("✅ OvR個別モデルレポートを保存しました: \(reportURL.path)")
        } catch {
            print("❌ OvR個別モデルレポートの保存エラー: \(error.localizedDescription) (Path: \(reportURL.path))")
        }
    }
}

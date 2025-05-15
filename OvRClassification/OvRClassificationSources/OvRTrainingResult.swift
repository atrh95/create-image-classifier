import CSInterface
import Foundation

public struct IndividualModelReport: Codable, Sendable {
    public let modelName: String
    public let positiveClassName: String
    public let trainingAccuracyRate: Double
    public let validationAccuracyRate: Double
    public let recallRate: Double
    public let precisionRate: Double
    public let modelDescription: String
}

public struct OvRTrainingResult: TrainingResultProtocol {
    public let modelName: String?
    public let modelOutputPath: String
    public let trainingDataPaths: String
    public let maxIterations: Int
    public let individualReports: [IndividualModelReport]

    public func saveLog(modelAuthor: String, modelDescription: String, modelVersion: String) {
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd HH:mm:ss Z"
        let generatedDateString = dateFormatter.string(from: Date())

        let reportFileName = "OvR_Run_Report_\(modelVersion).md"
        let modelDir = URL(fileURLWithPath: modelOutputPath)
        let reportURL = modelDir.appendingPathComponent(reportFileName)

        var markdownText = """
        # OvR (One-vs-Rest) トレーニング実行レポート

        ## 実行概要
        モデル群名         : \(modelName ?? "N/A")
        レポート生成日時   : \(generatedDateString)
        最大反復回数     : \(maxIterations) (各ペアモデル共通)
        """

        if !individualReports.isEmpty {
            markdownText += """

            ## 個別モデルのパフォーマンス指標
            | モデル名 (PositiveClass) | 検証正解率 | 検証再現率 | 検証適合率 | 説明 |
            |--------------------------|--------------|--------------|--------------|------|
            """
            for report in individualReports {
                let modelNameDisplay = "\(report.modelName) (\(report.positiveClassName))"
                let valAccStr = String(format: "%.2f%%", report.validationAccuracyRate * 100)
                let recallStr = String(format: "%.2f%%", report.recallRate * 100)
                let precisionStr = String(format: "%.2f%%", report.precisionRate * 100)

                let descSummary = report.modelDescription.prefix(50)

                markdownText +=
                    "\n| \(modelNameDisplay) | \(valAccStr) | \(recallStr) | \(precisionStr) | \(descSummary)... |"
            }
            markdownText += "\n"

            markdownText += "\n\n### 個別モデル詳細説明:\n"
            for report in individualReports {
                markdownText += "- **\(report.modelName) (\(report.positiveClassName))**: \(report.modelDescription)\n"
            }
        }

        markdownText += """

        ## 共通メタデータ
        作成者            : \(modelAuthor)
        全体説明          : \(modelDescription) (このOvR実行全体に対して)
        バージョン        : \(modelVersion)
        """

        do {
            try markdownText.write(to: reportURL, atomically: true, encoding: .utf8)
            print("✅ OvR実行レポートを保存しました: \(reportURL.path)")
        } catch {
            print("❌ OvR実行レポートの保存エラー: \(error.localizedDescription) (Path: \(reportURL.path))")
        }
    }
}

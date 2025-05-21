import CSInterface
import Foundation

public struct IndividualModelReport: Codable, Sendable {
    public let modelName: String
    public let positiveClassName: String
    public let trainingAccuracyRate: Double
    public let validationAccuracyPercentage: Double
    public let recallRate: Double
    public let precisionRate: Double
    public let modelDescription: String
    public let confusionMatrix: ConfusionMatrix
}

public struct ConfusionMatrix: Codable, Sendable {
    public let truePositive: Int
    public let falsePositive: Int
    public let falseNegative: Int
    public let trueNegative: Int
}

public struct OvRTrainingResult: TrainingResultProtocol {
    public let modelOutputPath: String
    public let trainingDataPaths: String
    public let maxIterations: Int
    public let individualReports: [IndividualModelReport]
    public let dataAugmentationDescription: String
    public let featureExtractorDescription: String

    public init(
        modelOutputPath: String,
        trainingDataPaths: String,
        maxIterations: Int,
        individualReports: [IndividualModelReport],
        dataAugmentationDescription: String,
        baseFeatureExtractorDescription: String,
        scenePrintRevision: Int?
    ) {
        self.modelOutputPath = modelOutputPath
        self.trainingDataPaths = trainingDataPaths
        self.maxIterations = maxIterations
        self.individualReports = individualReports
        self.dataAugmentationDescription = dataAugmentationDescription
        if let revision = scenePrintRevision {
            self.featureExtractorDescription = "\(baseFeatureExtractorDescription)(revision: \(revision))"
        } else {
            self.featureExtractorDescription = baseFeatureExtractorDescription
        }
    }

    public func saveLog(modelAuthor: String, modelName: String, modelVersion: String) {
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd HH:mm:ss Z"
        dateFormatter.timeZone = TimeZone(identifier: "Asia/Tokyo")
        let generatedDateString = dateFormatter.string(from: Date())

        let reportFileName = "OvR_Run_Report_\(modelVersion).md"
        let modelDir = URL(fileURLWithPath: modelOutputPath)
        let reportURL = modelDir.appendingPathComponent(reportFileName)

        var markdownText = """
        # OvR (One-vs-Rest) トレーニング実行レポート

        ## 実行概要
        モデル群         : OvRモデル群 (One-vs-Rest)
        モデルベース名   : \(modelName)
        レポート生成日時   : \(generatedDateString)
        最大反復回数     : \(maxIterations) (各ペアモデル共通)
        データ拡張       : \(dataAugmentationDescription)
        特徴抽出器       : \(featureExtractorDescription)

        ## 個別 "One" モデルのパフォーマンス指標
        | "One" クラス名 | 訓練正解率 | 検証正解率 | 再現率 | 適合率 |
        |----------------|--------------|--------------|----------|----------|
        """
        for report in individualReports {
            let trainAccStr = String(format: "%.2f%%", report.trainingAccuracyRate * 100)
            let valAccStr = String(format: "%.2f%%", report.validationAccuracyPercentage)
            let recallStr = String(format: "%.2f%%", report.recallRate * 100)
            let precisionStr = String(format: "%.2f%%", report.precisionRate * 100)
            markdownText += "\n| \(report.positiveClassName) | \(trainAccStr) | \(valAccStr) | \(recallStr) | \(precisionStr) |"
        }
        markdownText += "\n"

        // 混同行列の追加
        markdownText += """

        ## 混同行列（検証データ）
        """
        for report in individualReports {
            markdownText += """

            ### \(report.positiveClassName)
            ```
            +----------------+----------------+----------------+
            | True Label     | Predicted      | Count          |
            +----------------+----------------+----------------+
            | \(report.positiveClassName.padding(toLength: 14, withPad: " ", startingAt: 0)) | \(report.positiveClassName.padding(toLength: 14, withPad: " ", startingAt: 0)) | \(String(format: "%14d", report.confusionMatrix.truePositive)) |
            | \(report.positiveClassName.padding(toLength: 14, withPad: " ", startingAt: 0)) | Rest           | \(String(format: "%14d", report.confusionMatrix.falseNegative)) |
            | Rest           | \(report.positiveClassName.padding(toLength: 14, withPad: " ", startingAt: 0)) | \(String(format: "%14d", report.confusionMatrix.falsePositive)) |
            | Rest           | Rest           | \(String(format: "%14d", report.confusionMatrix.trueNegative)) |
            +----------------+----------------+----------------+
            ```
            """
        }

        markdownText += """

        ## 共通メタデータ
        作成者            : \(modelAuthor)
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

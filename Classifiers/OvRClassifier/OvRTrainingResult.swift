import CICConfusionMatrix
import CICInterface
import CICTrainingResult
import Foundation

public struct OvRTrainingResult: TrainingResultProtocol {
    public let metadata: CICTrainingMetadata
    public let individualModelReports: [CICIndividualModelReport]

    public var modelOutputPath: String {
        URL(fileURLWithPath: metadata.trainedModelFilePath).deletingLastPathComponent().path
    }

    public init(
        metadata: CICTrainingMetadata,
        individualModelReports: [CICIndividualModelReport]
    ) {
        self.metadata = metadata
        self.individualModelReports = individualModelReports
    }

    public func saveLog(
        modelAuthor: String,
        modelName: String,
        modelVersion: String
    ) {
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd HH:mm:ss Z"
        dateFormatter.timeZone = TimeZone(identifier: "Asia/Tokyo")
        let generatedDateString = dateFormatter.string(from: Date())

        var markdownText = """
        # モデルトレーニング情報: \(modelName)

        ## モデル詳細
        モデル名           : \(modelName)
        ファイル生成日時   : \(generatedDateString)
        最大反復回数     : \(metadata.maxIterations)
        データ拡張       : \(metadata.dataAugmentationDescription)
        特徴抽出器       : \(metadata.featureExtractorDescription)

        ## トレーニング設定
        使用されたクラスラベル : \(metadata.detectedClassLabelsList.joined(separator: ", "))

        """

        markdownText += """

        ## 個別モデルの性能指標
        | クラス | 訓練正解率 | 検証正解率 | 再現率 | 適合率 | F1スコア |
        |--------|------------|------------|--------|--------|----------|
        \(individualModelReports.enumerated().map { index, report in
            let recall = report.confusionMatrix?.recall ?? 0.0
            let precision = report.confusionMatrix?.precision ?? 0.0
            let f1Score = report.confusionMatrix?.f1Score ?? 0.0
            return "| \(String(format: "%2d", index + 1)) | \(report.positiveClassName) | \(String(format: "%.1f%%", report.trainingAccuracyRate * 100.0)) | \(String(format: "%.1f%%", report.validationAccuracyRate * 100.0)) | \(String(format: "%.1f%%", recall * 100.0)) | \(String(format: "%.1f%%", precision * 100.0)) | \(String(format: "%.3f", f1Score)) |"
        }.joined(separator: "\n"))

        ## モデルメタデータ
        作成者            : \(modelAuthor)
        バージョン          : \(modelVersion)
        """

        let outputDir = URL(fileURLWithPath: metadata.trainedModelFilePath).deletingLastPathComponent()
        let textFileName = "OvR_Run_Report_\(modelVersion).md"
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

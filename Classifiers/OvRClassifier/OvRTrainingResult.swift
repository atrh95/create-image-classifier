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
        \(individualModelReports.map { report in
            report.generateMarkdownReport()
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

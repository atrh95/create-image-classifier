import CICConfusionMatrix
import CICInterface
import CICTrainingResult
import Foundation

public struct MultiLabelTrainingResult: TrainingResultProtocol {
    public let metadata: CICTrainingMetadata
    public let metrics: (
        training: (accuracy: Double, errorRate: Double),
        validation: (accuracy: Double, errorRate: Double)
    )
    public let confusionMatrix: CICMultiClassConfusionMatrix?
    public let individualModelReports: [CICIndividualModelReport]

    public init(
        metadata: CICTrainingMetadata,
        metrics: (
            training: (accuracy: Double, errorRate: Double),
            validation: (accuracy: Double, errorRate: Double)
        ),
        confusionMatrix: CICMultiClassConfusionMatrix?,
        individualModelReports: [CICIndividualModelReport]
    ) {
        self.metadata = metadata
        self.metrics = metrics
        self.confusionMatrix = confusionMatrix
        self.individualModelReports = individualModelReports
    }

    public func saveLog(
        modelAuthor: String,
        modelName: String,
        modelVersion: String,
        outputDirPath: String
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
        使用されたラベル : \(metadata.classLabelCounts.map { "\($0.key) (\($0.value)枚)" }.joined(separator: ", "))
        """

        markdownText += """

        ## 個別モデルの性能指標
        | ラベル | 訓練正解率 | 検証正解率 | 再現率 | 適合率 | F1スコア |
        |--------|------------|------------|--------|--------|----------|
        \(individualModelReports.map { report in
            let trainingAccuracyPercent = report.metrics.training.accuracy * 100.0
            let validationAccuracyPercent = report.metrics.validation.accuracy * 100.0
            let recallPercent = report.confusionMatrix?.recall ?? 0.0 * 100.0
            let precisionPercent = report.confusionMatrix?.precision ?? 0.0 * 100.0
            let f1Score = report.confusionMatrix?.f1Score ?? 0.0
            return "| \(report.classCounts.positive.name) | \(String(format: "%.1f%%", trainingAccuracyPercent)) | \(String(format: "%.1f%%", validationAccuracyPercent)) | \(String(format: "%.1f%%", recallPercent)) | \(String(format: "%.1f%%", precisionPercent)) | \(String(format: "%.3f", f1Score)) |"
        }.joined(separator: "\n"))

        ## モデルメタデータ
        作成者            : \(modelAuthor)
        バージョン          : \(modelVersion)
        """

        let outputDir = URL(fileURLWithPath: outputDirPath)
        let textFileName = "\(modelName)_\(modelVersion).md"
        let textFilePath = outputDir.appendingPathComponent(textFileName).path

        do {
            try markdownText.write(toFile: textFilePath, atomically: true, encoding: String.Encoding.utf8)
            print("✅ [\(modelName)] モデル情報をMarkdownファイルに保存しました: \(textFilePath)")
        } catch {
            print("❌ [\(modelName)] Markdownファイルの書き込みに失敗しました: \(error.localizedDescription)")
        }
    }

    public func displayComparisonTable() {
        guard !individualModelReports.isEmpty else {
            print("\n📊 モデルの性能")
            print("+----------------------+-------+-------+-------+-------+-------+")
            print("| ラベル                | 訓練  | 検証  | 再現率 | 適合率 | F1    |")
            print("+----------------------+-------+-------+-------+-------+-------+")
            print("| データなし              | - | - | - | - | - |")
            print("+----------------------+-------+-------+-------+-------+-------+")
            return
        }
        
        print("\n📊 モデルの性能")
        print("+----------------------+-------+-------+-------+-------+-------+")
        print("| ラベル                | 訓練  | 検証  | 再現率 | 適合率 | F1    |")
        print("+----------------------+-------+-------+-------+-------+-------+")

        for report in individualModelReports {
            let label = String(report.classCounts.positive.name.prefix(20))
            let paddedLabel = label.padding(toLength: 20, withPad: " ", startingAt: 0)
            let trainingAccuracyPercent = report.metrics.training.accuracy * 100.0
            let validationAccuracyPercent = report.metrics.validation.accuracy * 100.0
            let recallPercent = (report.confusionMatrix?.recall ?? 0.0) * 100.0
            let precisionPercent = (report.confusionMatrix?.precision ?? 0.0) * 100.0
            let f1Score = report.confusionMatrix?.f1Score ?? 0.0
            
            print("| \(paddedLabel) | \(String(format: "%.1f", trainingAccuracyPercent))% | \(String(format: "%.1f", validationAccuracyPercent))% | \(String(format: "%.1f", recallPercent))% | \(String(format: "%.1f", precisionPercent))% | \(String(format: "%.3f", f1Score)) |")
        }
        print("+----------------------+-------+-------+-------+-------+-------+")
    }
}

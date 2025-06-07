import CICConfusionMatrix
import CICInterface
import CICTrainingResult
import Foundation

/// 画像分類モデルのトレーニング結果を格納する構造体
public struct BinaryTrainingResult: TrainingResultProtocol {
    public let metadata: CICTrainingMetadata
    public let metrics: (
        training: (accuracy: Double, errorRate: Double),
        validation: (accuracy: Double, errorRate: Double)
    )
    public let confusionMatrix: CICBinaryConfusionMatrix?
    public let individualModelReport: CICIndividualModelReport

    public init(
        metadata: CICTrainingMetadata,
        metrics: (
            training: (accuracy: Double, errorRate: Double),
            validation: (accuracy: Double, errorRate: Double)
        ),
        confusionMatrix: CICBinaryConfusionMatrix?,
        individualModelReport: CICIndividualModelReport
    ) {
        self.metadata = metadata
        self.metrics = metrics
        self.confusionMatrix = confusionMatrix
        self.individualModelReport = individualModelReport
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

        let trainingAccuracyPercent = String(format: "%.2f", metrics.training.accuracy * 100)
        let validationAccuracyPercent = String(format: "%.2f", metrics.validation.accuracy * 100)
        let trainingErrorPercent = String(format: "%.2f", metrics.training.errorRate * 100)
        let validationErrorPercent = String(format: "%.2f", metrics.validation.errorRate * 100)

        var markdownText = """
        # モデルトレーニング情報: \(modelName)

        ## モデル詳細
        モデル名           : \(modelName)
        ファイル生成日時   : \(generatedDateString)
        最大反復回数     : \(metadata.maxIterations)
        データ拡張       : \(metadata.dataAugmentationDescription)
        特徴抽出器       : \(metadata.featureExtractorDescription)

        ## トレーニング設定
        使用されたクラスラベル : \(metadata.classLabelCounts.map { "\($0.key) (\($0.value)枚)" }.joined(separator: ", "))

        ## パフォーマンス指標 (全体)
        トレーニング誤分類率 (学習時) : \(trainingErrorPercent)%
        訓練データ正解率 (学習時) : \(trainingAccuracyPercent)%
        検証データ正解率 (学習時自動検証) : \(validationAccuracyPercent)%
        検証誤分類率 (学習時自動検証) : \(validationErrorPercent)%

        """

        if let confusionMatrix {
            markdownText += """

            ## 性能指標
            | 指標 | 値 |
            |------|-----|
            | 再現率 | \(String(format: "%.1f%%", confusionMatrix.recall * 100.0)) |
            | 適合率 | \(String(format: "%.1f%%", confusionMatrix.precision * 100.0)) |
            | F1スコア | \(String(format: "%.3f", confusionMatrix.f1Score)) |

            """
        }

        markdownText += """

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
        guard let confusionMatrix else { return }

        print("\n📊 モデルの性能")
        print(
            "+------------------+------------------+------------------+------------------+------------------+"
        )
        print("| 訓練正解率       | 検証正解率       | 再現率           | 適合率           | F1スコア         |")
        print(
            "+------------------+------------------+------------------+------------------+------------------+"
        )

        let trainingAccuracyPercent = metrics.training.accuracy * 100.0
        let validationAccuracyPercent = metrics.validation.accuracy * 100.0
        let recallPercent = confusionMatrix.recall * 100.0
        let precisionPercent = confusionMatrix.precision * 100.0
        print(
            "| \(String(format: "%14.1f%%", trainingAccuracyPercent)) | \(String(format: "%14.1f%%", validationAccuracyPercent)) | \(String(format: "%14.1f%%", recallPercent)) | \(String(format: "%14.1f%%", precisionPercent)) | \(String(format: "%14.3f", confusionMatrix.f1Score)) |"
        )
        print(
            "+------------------+------------------+------------------+------------------+------------------+"
        )
    }
}

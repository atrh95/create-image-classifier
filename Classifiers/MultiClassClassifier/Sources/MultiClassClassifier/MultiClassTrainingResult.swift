import CICConfusionMatrix
import CICInterface
import CICTrainingResult
import Foundation

public struct MultiClassTrainingResult: TrainingResultProtocol {
    // メタデータ
    public let metadata: CICTrainingMetadata

    // パフォーマンス指標
    public let metrics: (
        training: (accuracy: Double, errorRate: Double),
        validation: (accuracy: Double, errorRate: Double)
    )

    // 詳細な性能指標
    public let confusionMatrix: CICMultiClassConfusionMatrix?
    public let classMetrics: [ClassMetrics]

    public init(
        metadata: CICTrainingMetadata,
        metrics: (
            training: (accuracy: Double, errorRate: Double),
            validation: (accuracy: Double, errorRate: Double)
        ),
        confusionMatrix: CICMultiClassConfusionMatrix?
    ) {
        self.metadata = metadata
        self.metrics = metrics
        self.confusionMatrix = confusionMatrix
        classMetrics = confusionMatrix?.calculateMetrics() ?? []
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

        let trainingAccStr = String(format: "%.2f", metrics.training.accuracy)
        let validationAccStr = String(format: "%.2f", metrics.validation.accuracy)
        let trainingErrStr = String(format: "%.2f", metrics.training.errorRate * 100)
        let validationErrStr = String(format: "%.2f", metrics.validation.errorRate * 100)

        var markdownText = """
        # モデルトレーニング情報: \(modelName)

        ## モデル詳細
        モデル名           : \(modelName)
        ファイル生成日時   : \(generatedDateString)
        最大反復回数     : \(metadata.maxIterations)
        データ拡張       : \(metadata.dataAugmentationDescription)
        特徴抽出器       : \(metadata.featureExtractorDescription)

        ## トレーニング設定
        使用されたクラス : \(metadata.classLabelCounts.map { "\($0.key) (\($0.value)枚)" }.joined(separator: ", "))

        ## パフォーマンス指標 (全体)
        トレーニング誤分類率 (学習時) : \(trainingErrStr)%
        訓練データ正解率 (学習時) : \(trainingAccStr)%
        検証データ正解率 (学習時自動検証) : \(validationAccStr)%
        検証誤分類率 (学習時自動検証) : \(validationErrStr)%

        """

        if confusionMatrix != nil {
            let classMetrics = confusionMatrix?.calculateMetrics() ?? []
            markdownText += """

            ## クラス別性能指標
            | クラス | 再現率 | 適合率 | F1スコア |
            |:---|:---|:---|:---|
            \(classMetrics.isEmpty ? "" : classMetrics.map { metric in
                "| \(metric.label) | \(String(format: "%.1f", metric.recall * 100.0))% | \(String(format: "%.1f", metric.precision * 100.0))% | \(String(format: "%.3f", metric.f1Score)) |"
            }.joined(separator: "\n"))
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
        guard confusionMatrix != nil else { return }

        print("\n📊 モデルの性能")
        print("+----------------------+-------+-------+-------+-------+-------+")
        print("| クラス                | 訓練  | 検証  | 再現率 | 適合率 | F1    |")
        print("+----------------------+-------+-------+-------+-------+-------+")

        guard !classMetrics.isEmpty else {
            print("| データなし              | - | - | - | - | - |")
            print("+----------------------+-------+-------+-------+-------+-------+")
            return
        }

        for metric in classMetrics {
            let label = String(metric.label.prefix(20))
            let paddedLabel = label.padding(toLength: 20, withPad: " ", startingAt: 0)
            let trainingAcc = String(format: "%.1f", metrics.training.accuracy * 100.0)
            let validationAcc = String(format: "%.1f", metrics.validation.accuracy * 100.0)
            let recall = String(format: "%.1f", metric.recall * 100.0)
            let precision = String(format: "%.1f", metric.precision * 100.0)
            let f1Score = String(format: "%.3f", metric.f1Score)

            print("| \(paddedLabel) | \(trainingAcc)% | \(validationAcc)% | \(recall)% | \(precision)% | \(f1Score) |")
        }
        print("+----------------------+-------+-------+-------+-------+-------+")
    }
}

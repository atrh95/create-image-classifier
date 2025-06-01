import CICConfusionMatrix
import Foundation

/// 個別モデルのトレーニング結果を格納する構造体
public struct CICIndividualModelReport {
    public let modelFileName: String
    public let metrics: (
        training: (accuracy: Double, errorRate: Double),
        validation: (accuracy: Double, errorRate: Double)
    )
    public let confusionMatrix: CICBinaryConfusionMatrix?
    public let classCounts: (positive: (name: String, count: Int), negative: (name: String, count: Int))

    public init(
        modelFileName: String,
        metrics: (
            training: (accuracy: Double, errorRate: Double),
            validation: (accuracy: Double, errorRate: Double)
        ),
        confusionMatrix: CICBinaryConfusionMatrix?,
        classCounts: (positive: (name: String, count: Int), negative: (name: String, count: Int))
    ) {
        self.modelFileName = modelFileName
        self.metrics = metrics
        self.confusionMatrix = confusionMatrix
        self.classCounts = classCounts
    }

    /// 個別モデルのレポートをMarkdown形式で生成
    public func generateMarkdownReport() -> String {
        var report = """
        ## \(classCounts.positive.name)
        - 訓練正解率: \(String(format: "%.1f%%", metrics.training.accuracy * 100.0))
        - 検証正解率: \(String(format: "%.1f%%", metrics.validation.accuracy * 100.0))
        """

        if let confusionMatrix {
            report += """

            - 再現率 (Recall)    : \(String(format: "%.1f%%", confusionMatrix.recall * 100.0))
            - 適合率 (Precision) : \(String(format: "%.1f%%", confusionMatrix.precision * 100.0))
            - F1スコア          : \(String(format: "%.3f", confusionMatrix.f1Score))
            """
        } else {
            report += "\n⚠️ 検証データが不十分なため、混同行列の計算をスキップしました\n"
        }

        return report
    }
}

import CICConfusionMatrix
import CICInterface
import Foundation

/// 多クラス分類モデルのトレーニング結果を格納する構造体
public struct CICMultiClassModelReport {
    public let modelFileName: String
    public let metrics: (
        training: (accuracy: Double, errorRate: Double),
        validation: (accuracy: Double, errorRate: Double)
    )
    public let confusionMatrix: CICMultiClassConfusionMatrix?
    public let classCounts: [String: Int]

    public init(
        modelFileName: String,
        metrics: (
            training: (accuracy: Double, errorRate: Double),
            validation: (accuracy: Double, errorRate: Double)
        ),
        confusionMatrix: CICMultiClassConfusionMatrix?,
        classCounts: [String: Int]
    ) {
        self.modelFileName = modelFileName
        self.metrics = metrics
        self.confusionMatrix = confusionMatrix
        self.classCounts = classCounts
    }

    public func generateMetricsDescription() -> String {
        var description = """
        クラス: \(classCounts.map { "\($0.key): \($0.value)枚" }.joined(separator: ", "))
        訓練正解率: \(String(format: "%.1f%%", metrics.training.accuracy * 100.0))
        検証正解率: \(String(format: "%.1f%%", metrics.validation.accuracy * 100.0))
        """

        if let confusionMatrix {
            let classMetrics = confusionMatrix.calculateMetrics()
            description += "\n\nクラス別性能指標:\n"
            for metric in classMetrics {
                description += """

                \(metric.label):
                再現率: \(String(format: "%.1f%%", metric.recall * 100.0))
                適合率: \(String(format: "%.1f%%", metric.precision * 100.0))
                F1スコア: \(String(format: "%.3f", metric.f1Score))
                """
            }
        }

        return description
    }
}

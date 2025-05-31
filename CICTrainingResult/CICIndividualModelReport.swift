import CICConfusionMatrix
import Foundation

/// 個別モデルのトレーニング結果を格納する構造体
public struct CICIndividualModelReport {
    public let modelName: String
    public let positiveClassName: String
    public let trainingAccuracyRate: Double
    public let validationAccuracyRate: Double
    public let confusionMatrix: CICBinaryConfusionMatrix?

    public init(
        modelName: String,
        positiveClassName: String,
        trainingAccuracyRate: Double,
        validationAccuracyRate: Double,
        confusionMatrix: CICBinaryConfusionMatrix?
    ) {
        self.modelName = modelName
        self.positiveClassName = positiveClassName
        self.trainingAccuracyRate = trainingAccuracyRate
        self.validationAccuracyRate = validationAccuracyRate
        self.confusionMatrix = confusionMatrix
    }

    /// 個別モデルのレポートをMarkdown形式で生成
    public func generateMarkdownReport() -> String {
        var report = """
        ## \(positiveClassName)
        - 訓練正解率: \(String(format: "%.1f%%", trainingAccuracyRate * 100.0))
        - 検証正解率: \(String(format: "%.1f%%", validationAccuracyRate * 100.0))
        """

        if let confusionMatrix {
            report += """

            - 再現率 (Recall)    : \(String(format: "%.1f%%", confusionMatrix.recall * 100.0))
            - 適合率 (Precision) : \(String(format: "%.1f%%", confusionMatrix.precision * 100.0))
            - F1スコア          : \(String(format: "%.1f%%", confusionMatrix.f1Score * 100.0))

            \(confusionMatrix.getMatrixGraph())
            """
        } else {
            report += "\n⚠️ 検証データが不十分なため、混同行列の計算をスキップしました\n"
        }

        return report
    }
}

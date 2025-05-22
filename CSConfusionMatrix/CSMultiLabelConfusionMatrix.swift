import Foundation

public struct LabelMetrics {
    public let label: String
    public var truePositives: Int
    public var falsePositives: Int
    public var falseNegatives: Int

    public var recall: Double? {
        let denominator = truePositives + falseNegatives
        return denominator == 0 ? nil : Double(truePositives) / Double(denominator)
    }

    public var precision: Double? {
        let denominator = truePositives + falsePositives
        return denominator == 0 ? nil : Double(truePositives) / Double(denominator)
    }

    public var f1Score: Double? {
        guard let precision,
              let recall else { return nil }
        let denominator = precision + recall
        return denominator == 0 ? nil : 2 * precision * recall / denominator
    }
}

public struct CSMultiLabelConfusionMatrix {
    private let perLabelMetrics: [String: LabelMetrics]
    private let labels: [String]
    private let predictionThreshold: Float

    public init(
        predictions: [(trueLabels: Set<String>, predictedLabels: Set<String>)],
        labels: [String],
        predictionThreshold: Float = 0.5
    ) {
        self.labels = labels
        self.predictionThreshold = predictionThreshold

        var metrics: [String: LabelMetrics] = [:]
        for label in labels {
            metrics[label] = LabelMetrics(
                label: label,
                truePositives: 0,
                falsePositives: 0,
                falseNegatives: 0
            )
        }

        for (trueLabels, predictedLabels) in predictions {
            for label in labels {
                let trulyHasLabel = trueLabels.contains(label)
                let predictedHasLabel = predictedLabels.contains(label)

                if trulyHasLabel, predictedHasLabel {
                    metrics[label]?.truePositives += 1
                } else if !trulyHasLabel, predictedHasLabel {
                    metrics[label]?.falsePositives += 1
                } else if trulyHasLabel, !predictedHasLabel {
                    metrics[label]?.falseNegatives += 1
                }
            }
        }

        perLabelMetrics = metrics
    }

    public func getMatrixGraph() -> String {
        var result = ""

        // ヘッダー
        result += "Label\tTrue Positives\tTotal Actual\n"

        // データ行
        let metrics = calculateMetrics()
        for metric in metrics {
            result += "\(metric.label)\t\(metric.truePositives)\t\(metric.truePositives + metric.falseNegatives)\n"
        }

        return result
    }

    public func calculateMetrics() -> [LabelMetrics] {
        labels.sorted().compactMap { perLabelMetrics[$0] }
    }
}

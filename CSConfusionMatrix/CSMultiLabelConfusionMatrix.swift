import Foundation

public struct CSMultiLabelConfusionMatrix {
    private let perLabelMetrics: [String: (tp: Int, fp: Int, fn: Int)]
    private let labels: [String]
    private let predictionThreshold: Float

    public init(
        predictions: [(trueLabels: Set<String>, predictedLabels: Set<String>)],
        labels: [String],
        predictionThreshold: Float = 0.5
    ) {
        self.labels = labels
        self.predictionThreshold = predictionThreshold

        var metrics: [String: (tp: Int, fp: Int, fn: Int)] = [:]
        for label in labels {
            metrics[label] = (tp: 0, fp: 0, fn: 0)
        }

        for (trueLabels, predictedLabels) in predictions {
            for label in labels {
                let trulyHasLabel = trueLabels.contains(label)
                let predictedHasLabel = predictedLabels.contains(label)

                if trulyHasLabel, predictedHasLabel {
                    metrics[label]?.tp += 1
                } else if !trulyHasLabel, predictedHasLabel {
                    metrics[label]?.fp += 1
                } else if trulyHasLabel, !predictedHasLabel {
                    metrics[label]?.fn += 1
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
        var metrics: [LabelMetrics] = []

        for label in labels.sorted() {
            if let counts = perLabelMetrics[label] {
                let recall = (counts.tp + counts.fn == 0) ? nil : Double(counts.tp) / Double(counts.tp + counts.fn)
                let precision = (counts.tp + counts.fp == 0) ? nil : Double(counts.tp) / Double(counts.tp + counts.fp)
                let f1Score: Double? = if let p = precision, let r = recall, p + r != 0 {
                    2 * p * r / (p + r)
                } else {
                    nil
                }

                metrics.append(LabelMetrics(
                    label: label,
                    recall: recall,
                    precision: precision,
                    f1Score: f1Score,
                    truePositives: counts.tp,
                    falsePositives: counts.fp,
                    falseNegatives: counts.fn
                ))
            }
        }

        return metrics
    }

    public func getAverageMetrics() -> (recall: Double, precision: Double, f1Score: Double)? {
        let metrics = calculateMetrics()
        guard !metrics.isEmpty else { return nil }

        let validMetrics = metrics.filter { $0.recall != nil && $0.precision != nil && $0.f1Score != nil }
        guard !validMetrics.isEmpty else { return nil }

        let avgRecall = validMetrics.map(\.recall!).reduce(0, +) / Double(validMetrics.count)
        let avgPrecision = validMetrics.map(\.precision!).reduce(0, +) / Double(validMetrics.count)
        let avgF1Score = validMetrics.map(\.f1Score!).reduce(0, +) / Double(validMetrics.count)

        return (recall: avgRecall, precision: avgPrecision, f1Score: avgF1Score)
    }
}

public struct LabelMetrics {
    public let label: String
    public let recall: Double?
    public let precision: Double?
    public let f1Score: Double?
    public let truePositives: Int
    public let falsePositives: Int
    public let falseNegatives: Int
}

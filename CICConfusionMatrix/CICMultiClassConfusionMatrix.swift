import CreateML
import Foundation

public struct ClassMetrics {
    public let label: String
    public let recall: Double
    public let precision: Double
    public let f1Score: Double
}

public final class CICMultiClassConfusionMatrix {
    private let dataTable: MLDataTable
    private let predictedColumn: String
    private let actualColumn: String

    public let labels: [String]
    public private(set) var matrix: [[Int]]

    public static func validateDataTable(
        _ dataTable: MLDataTable,
        predictedColumn: String,
        actualColumn: String
    ) -> Bool {
        // データの有効性チェック
        guard !dataTable.rows.isEmpty else {
            print("❌ エラー: データテーブルが空です")
            return false
        }

        guard dataTable.columnNames.contains(predictedColumn) else {
            print("❌ エラー: 予測列 '\(predictedColumn)' が存在しません")
            print("   利用可能な列: \(dataTable.columnNames.joined(separator: ", "))")
            return false
        }

        guard dataTable.columnNames.contains(actualColumn) else {
            print("❌ エラー: 実際の値の列 '\(actualColumn)' が存在しません")
            print("   利用可能な列: \(dataTable.columnNames.joined(separator: ", "))")
            return false
        }

        // 予測値と実際の値のラベルセットを取得
        let predictedLabels = Set(dataTable.rows.compactMap { $0[predictedColumn]?.stringValue })
        let actualLabels = Set(dataTable.rows.compactMap { $0[actualColumn]?.stringValue })

        // ラベルが存在することを確認
        guard !predictedLabels.isEmpty, !actualLabels.isEmpty else {
            print("❌ エラー: ラベルが存在しません")
            print("   予測ラベル: \(predictedLabels.joined(separator: ", "))")
            print("   実際のラベル: \(actualLabels.joined(separator: ", "))")
            return false
        }

        // 予測値と実際の値のラベルセットが一致することを確認
        guard predictedLabels == actualLabels else {
            print("❌ エラー: 予測ラベルと実際のラベルが一致しません")
            print("   予測ラベル: \(predictedLabels.joined(separator: ", "))")
            print("   実際のラベル: \(actualLabels.joined(separator: ", "))")
            return false
        }

        return true
    }

    public init?(dataTable: MLDataTable, predictedColumn: String, actualColumn: String) {
        self.dataTable = dataTable
        self.predictedColumn = predictedColumn
        self.actualColumn = actualColumn

        // データの有効性チェック
        guard Self.validateDataTable(dataTable, predictedColumn: predictedColumn, actualColumn: actualColumn) else {
            return nil
        }

        // ラベルの取得とソート
        let labelSet = Set(dataTable.rows.compactMap { $0[actualColumn]?.stringValue })
        labels = Array(labelSet).sorted()

        // 混同行列の初期化
        let size = labels.count
        matrix = Array(repeating: Array(repeating: 0, count: size), count: size)

        // ラベルからインデックスへのマッピング
        let labelToIndex = Dictionary(uniqueKeysWithValues: labels.enumerated().map { ($1, $0) })

        // 混同行列の計算
        calculateConfusionMatrix(labelToIndex: labelToIndex)
    }

    private func calculateConfusionMatrix(labelToIndex: [String: Int]) {
        for row in dataTable.rows {
            guard let actualLabel = row[actualColumn]?.stringValue,
                  let predictedLabel = row[predictedColumn]?.stringValue,
                  let actualIndex = labelToIndex[actualLabel],
                  let predictedIndex = labelToIndex[predictedLabel],
                  let count = row["Count"]?.intValue
            else {
                continue
            }
            matrix[actualIndex][predictedIndex] += count
        }
    }

    public func calculateMetrics() -> [ClassMetrics] {
        labels.enumerated().map { index, label in
            let row = matrix[index]
            let column = matrix.map { $0[index] }

            let truePositives = row[index]
            let falsePositives = column.reduce(0, +) - truePositives
            let falseNegatives = row.reduce(0, +) - truePositives

            let recall = calculateRecall(truePositives: truePositives, falseNegatives: falseNegatives)
            let precision = calculatePrecision(truePositives: truePositives, falsePositives: falsePositives)
            let f1Score = calculateF1Score(recall: recall, precision: precision)

            return ClassMetrics(
                label: label,
                recall: recall,
                precision: precision,
                f1Score: f1Score
            )
        }
    }

    private func calculateRecall(truePositives: Int, falseNegatives: Int) -> Double {
        let denominator = truePositives + falseNegatives
        return denominator == 0 ? 0.0 : Double(truePositives) / Double(denominator)
    }

    private func calculatePrecision(truePositives: Int, falsePositives: Int) -> Double {
        let denominator = truePositives + falsePositives
        return denominator == 0 ? 0.0 : Double(truePositives) / Double(denominator)
    }

    private func calculateF1Score(recall: Double, precision: Double) -> Double {
        let denominator = recall + precision
        return denominator == 0 ? 0.0 : 2 * (recall * precision) / denominator
    }

    public func getMatrixGraph() -> String {
        var result = ""

        // ヘッダー
        result += "Actual\\Predicted"
        for label in labels {
            result += " | \(label)"
        }
        result += "\n"

        // 各行
        for (i, label) in labels.enumerated() {
            result += label
            for value in matrix[i] {
                result += " | \(value)"
            }
            result += "\n"
        }

        return result
    }
}

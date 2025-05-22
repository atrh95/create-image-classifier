import CSInterface
import CreateML
import Foundation

public final class CSMultiClassConfusionMatrix: CSMultiClassConfusionMatrixProtocol {
    private let dataTable: MLDataTable
    private let predictedColumn: String
    private let actualColumn: String
    
    public let labels: [String]
    public private(set) var matrix: [[Int]]
    
    static func validateDataTable(
        _ dataTable: MLDataTable,
        predictedColumn: String,
        actualColumn: String
    ) -> Bool {
        // データの有効性チェック
        guard !dataTable.rows.isEmpty,
              dataTable.columnNames.contains(predictedColumn),
              dataTable.columnNames.contains(actualColumn) else {
            return false
        }
        
        // 予測値と実際の値のラベルセットを取得
        let predictedLabels = Set(dataTable.rows.compactMap { $0[predictedColumn]?.stringValue })
        let actualLabels = Set(dataTable.rows.compactMap { $0[actualColumn]?.stringValue })
        
        // ラベルが存在することを確認
        guard !predictedLabels.isEmpty, !actualLabels.isEmpty else {
            return false
        }
        
        // 予測値と実際の値のラベルセットが一致することを確認
        guard predictedLabels == actualLabels else {
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
        self.labels = Array(labelSet).sorted()
        
        // 混同行列の初期化
        let size = self.labels.count
        self.matrix = Array(repeating: Array(repeating: 0, count: size), count: size)
        
        // ラベルからインデックスへのマッピング
        let labelToIndex = Dictionary(uniqueKeysWithValues: self.labels.enumerated().map { ($1, $0) })
        
        // 混同行列の計算
        for row in dataTable.rows {
            guard let actualLabel = row[actualColumn]?.stringValue,
                  let predictedLabel = row[predictedColumn]?.stringValue,
                  let actualIndex = labelToIndex[actualLabel],
                  let predictedIndex = labelToIndex[predictedLabel] else {
                continue
            }
            self.matrix[actualIndex][predictedIndex] += 1
        }
    }
    
    public func calculateMetrics() -> [(label: String, recall: Double, precision: Double, f1Score: Double)] {
        return labels.enumerated().map { index, label in
            let row = matrix[index]
            let column = matrix.map { $0[index] }
            
            let truePositives = row[index]
            let falsePositives = column.reduce(0, +) - truePositives
            let falseNegatives = row.reduce(0, +) - truePositives
            
            let recall = calculateRecall(truePositives: truePositives, falseNegatives: falseNegatives)
            let precision = calculatePrecision(truePositives: truePositives, falsePositives: falsePositives)
            let f1Score = calculateF1Score(recall: recall, precision: precision)
            
            return (label: label, recall: recall, precision: precision, f1Score: f1Score)
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
    
    public func printMatrix() {
        // ヘッダー行の印刷
        print("\nConfusion Matrix:")
        print("Actual \\ Predicted", terminator: "\t")
        for label in labels {
            print(label, terminator: "\t")
        }
        print()
        
        // 各行の印刷
        for (i, label) in labels.enumerated() {
            print(label, terminator: "\t")
            for value in matrix[i] {
                print(value, terminator: "\t")
            }
            print()
        }
        print()
    }
} 

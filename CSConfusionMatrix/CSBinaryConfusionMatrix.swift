import CSInterface
import CreateML
import Foundation

public final class CSBinaryConfusionMatrix: CSBinaryConfusionMatrixProtocol {
    private let dataTable: MLDataTable
    private let predictedColumn: String
    private let actualColumn: String
    
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
        
        // ラベルの取得とソート
        let labelSet = Set(dataTable.rows.compactMap { $0[actualColumn]?.stringValue })
        let sortedLabels = Array(labelSet).sorted()
        
        // 2クラス分類の確認
        guard sortedLabels.count == 2 else {
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
        let sortedLabels = Array(labelSet).sorted()
        
        // 混同行列の初期化
        self.matrix = Array(repeating: Array(repeating: 0, count: 2), count: 2)
        
        // ラベルからインデックスへのマッピング
        let labelToIndex = Dictionary(uniqueKeysWithValues: sortedLabels.enumerated().map { ($1, $0) })
        
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
    
    public var truePositive: Int { matrix[1][1] }
    public var falsePositive: Int { matrix[0][1] }
    public var falseNegative: Int { matrix[1][0] }
    public var trueNegative: Int { matrix[0][0] }
    
    public var recall: Double {
        calculateRecall(truePositives: truePositive, falseNegatives: falseNegative)
    }
    
    public var precision: Double {
        calculatePrecision(truePositives: truePositive, falsePositives: falsePositive)
    }
    
    public var accuracy: Double {
        let total = truePositive + falsePositive + falseNegative + trueNegative
        return total == 0 ? 0.0 : Double(truePositive + trueNegative) / Double(total)
    }
    
    public var f1Score: Double {
        calculateF1Score(recall: recall, precision: precision)
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
        print("Negative", terminator: "\t")
        print("Positive")
        
        // 各行の印刷
        print("Negative", terminator: "\t")
        print(matrix[0][0], terminator: "\t")
        print(matrix[0][1])
        
        print("Positive", terminator: "\t")
        print(matrix[1][0], terminator: "\t")
        print(matrix[1][1])
        print()
    }
} 

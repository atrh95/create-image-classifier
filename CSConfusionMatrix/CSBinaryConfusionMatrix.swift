import CSInterface
import CreateML

public struct CSBinaryConfusionMatrix: CSBinaryConfusionMatrixProtocol {
    public let truePositive: Int
    public let falsePositive: Int
    public let falseNegative: Int
    public let trueNegative: Int
    
    public init(dataTable: MLDataTable, predictedColumn: String, actualColumn: String) {
        // 必要なカラムが存在するか確認
        guard dataTable.columnNames.contains(predictedColumn),
              dataTable.columnNames.contains(actualColumn) else {
            self.truePositive = 0
            self.falsePositive = 0
            self.falseNegative = 0
            self.trueNegative = 0
            return
        }
        
        var labelSet = Set<String>()
        for row in dataTable.rows {
            if let actual = row[actualColumn]?.stringValue { labelSet.insert(actual) }
            if let predicted = row[predictedColumn]?.stringValue { labelSet.insert(predicted) }
        }
        
        // ラベルが2つあることを確認
        guard labelSet.count == 2 else {
            self.truePositive = 0
            self.falsePositive = 0
            self.falseNegative = 0
            self.trueNegative = 0
            return
        }
        
        let labels = Array(labelSet).sorted()
        let positiveLabel = labels[1]
        let negativeLabel = labels[0]
        
        var tp = 0
        var fp = 0
        var fn = 0
        var tn = 0
        
        for row in dataTable.rows {
            guard
                let actual = row[actualColumn]?.stringValue,
                let predicted = row[predictedColumn]?.stringValue
            else { continue }
            
            if actual == positiveLabel && predicted == positiveLabel {
                tp += 1
            } else if actual == negativeLabel && predicted == positiveLabel {
                fp += 1
            } else if actual == positiveLabel && predicted == negativeLabel {
                fn += 1
            } else if actual == negativeLabel && predicted == negativeLabel {
                tn += 1
            }
        }
        
        self.truePositive = tp
        self.falsePositive = fp
        self.falseNegative = fn
        self.trueNegative = tn
    }
    
    public var recall: Double {
        let denominator = Double(truePositive + falseNegative)
        return denominator > 0 ? Double(truePositive) / denominator : 0.0
    }
    
    public var precision: Double {
        let denominator = Double(truePositive + falsePositive)
        return denominator > 0 ? Double(truePositive) / denominator : 0.0
    }
    
    public var accuracy: Double {
        let total = Double(truePositive + falsePositive + falseNegative + trueNegative)
        return total > 0 ? Double(truePositive + trueNegative) / total : 0.0
    }
    
    public var f1Score: Double {
        let denominator = precision + recall
        return denominator > 0 ? 2 * (precision * recall) / denominator : 0.0
    }
    
    public func printMatrix(label: String? = nil) {
        let labelWidth = label?.count ?? 0
        let maxWidth = max(labelWidth, 8)
        
        print("\n📊 混同行列")
        print("  ┌" + String(repeating: "─", count: maxWidth + 2) + "┬" + String(repeating: "─", count: 8) + "┬" + String(repeating: "─", count: 8) + "┐")
        print("  │" + String(repeating: " ", count: maxWidth + 2) + "│" + " 予測値 ".padding(toLength: 8, withPad: " ", startingAt: 0) + "│" + " 実際値 ".padding(toLength: 8, withPad: " ", startingAt: 0) + "│")
        print("  ├" + String(repeating: "─", count: maxWidth + 2) + "┼" + String(repeating: "─", count: 8) + "┼" + String(repeating: "─", count: 8) + "┤")
        
        if let label = label {
            print(String(format: "  │ %-\(maxWidth)s │ %6d │ %6d │",
                label,
                truePositive,
                truePositive + falseNegative))
        } else {
            print(String(format: "  │ %-\(maxWidth)s │ %6d │ %6d │",
                "陽性",
                truePositive,
                truePositive + falseNegative))
        }
        
        print("  └" + String(repeating: "─", count: maxWidth + 2) + "┴" + String(repeating: "─", count: 8) + "┴" + String(repeating: "─", count: 8) + "┘")
        
        print(String(format: "  再現率: %.1f%%, 適合率: %.1f%%, F1スコア: %.1f%%",
            recall * 100,
            precision * 100,
            f1Score * 100))
    }
} 

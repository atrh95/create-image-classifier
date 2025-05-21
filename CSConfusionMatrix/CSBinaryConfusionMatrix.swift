import CSInterface

public struct CSBinaryConfusionMatrix: CSBinaryConfusionMatrixProtocol {
    public let truePositive: Int
    public let falsePositive: Int
    public let falseNegative: Int
    public let trueNegative: Int
    
    public init(truePositive: Int, falsePositive: Int, falseNegative: Int, trueNegative: Int) {
        self.truePositive = truePositive
        self.falsePositive = falsePositive
        self.falseNegative = falseNegative
        self.trueNegative = trueNegative
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

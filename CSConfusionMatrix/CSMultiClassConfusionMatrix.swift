import CSInterface

public struct CSMultiClassConfusionMatrix: CSMultiClassConfusionMatrixProtocol {
    public let matrix: [[Int]]
    public let labels: [String]
    
    public init(matrix: [[Int]], labels: [String]) {
        self.matrix = matrix
        self.labels = labels
    }
    
    public func printMatrix() {
        let maxLabelLength = labels.map { $0.count }.max() ?? 0
        let labelWidth = max(maxLabelLength, 8)
        
        print("\n📊 混同行列")
        print("  ┌" + String(repeating: "─", count: labelWidth + 2) + "┬" + String(repeating: "─", count: 8) + "┬" + String(repeating: "─", count: 8) + "┐")
        print("  │" + String(repeating: " ", count: labelWidth + 2) + "│" + " 予測値 ".padding(toLength: 8, withPad: " ", startingAt: 0) + "│" + " 実際値 ".padding(toLength: 8, withPad: " ", startingAt: 0) + "│")
        print("  ├" + String(repeating: "─", count: labelWidth + 2) + "┼" + String(repeating: "─", count: 8) + "┼" + String(repeating: "─", count: 8) + "┤")
        
        for (i, label) in labels.enumerated() {
            let rowSum = matrix[i].reduce(0, +)
            print(String(format: "  │ %-\(labelWidth)s │ %6d │ %6d │",
                label,
                matrix[i][i],
                rowSum))
        }
        print("  └" + String(repeating: "─", count: labelWidth + 2) + "┴" + String(repeating: "─", count: 8) + "┴" + String(repeating: "─", count: 8) + "┘")
    }
    
    public func calculateMetrics() -> [(label: String, recall: Double, precision: Double)] {
        var metrics: [(label: String, recall: Double, precision: Double)] = []
        
        for (i, label) in labels.enumerated() {
            var truePositives = matrix[i][i]
            var falsePositives = 0
            var falseNegatives = 0
            
            // 列の合計（予測値）から真陽性を引く
            for row in matrix {
                falsePositives += row[i]
            }
            falsePositives -= truePositives
            
            // 行の合計（実際値）から真陽性を引く
            falseNegatives = matrix[i].reduce(0, +) - truePositives
            
            let recall = (truePositives + falseNegatives) > 0 ? Double(truePositives) / Double(truePositives + falseNegatives) : 0.0
            let precision = (truePositives + falsePositives) > 0 ? Double(truePositives) / Double(truePositives + falsePositives) : 0.0
            
            metrics.append((label: label, recall: recall, precision: precision))
        }
        
        return metrics
    }
} 

import CSInterface
import CreateML

public struct CSMultiClassConfusionMatrix: CSMultiClassConfusionMatrixProtocol {
    public let matrix: [[Int]]
    public let labels: [String]
    private let detailedMetrics: [(label: String, recall: Double, precision: Double, f1Score: Double)]
    
    public init(dataTable: MLDataTable, predictedColumn: String, actualColumn: String) {
        // 必要なカラムが存在するか確認
        guard dataTable.columnNames.contains(predictedColumn),
              dataTable.columnNames.contains(actualColumn) else {
            self.matrix = []
            self.labels = []
            self.detailedMetrics = []
            return
        }
        
        // Get unique labels from both predicted and actual columns
        var labelSet = Set<String>()
        for row in dataTable.rows {
            if let actual = row[actualColumn]?.stringValue { labelSet.insert(actual) }
            if let predicted = row[predictedColumn]?.stringValue { labelSet.insert(predicted) }
        }
        
        // 少なくとも2つのクラスがあることを確認
        guard labelSet.count >= 2 else {
            self.matrix = []
            self.labels = []
            self.detailedMetrics = []
            return
        }
        
        self.labels = Array(labelSet).sorted()
        
        // Initialize confusion matrix with zeros
        var confusionMatrix = Array(repeating: Array(repeating: 0, count: labels.count), count: labels.count)
        
        // Fill the confusion matrix
        for row in dataTable.rows {
            guard
                let actual = row[actualColumn]?.stringValue,
                let predicted = row[predictedColumn]?.stringValue,
                let actualIndex = labels.firstIndex(of: actual),
                let predictedIndex = labels.firstIndex(of: predicted)
            else { continue }
            
            confusionMatrix[actualIndex][predictedIndex] += 1
        }
        
        self.matrix = confusionMatrix
        
        // Calculate detailed metrics for each class
        var metrics: [(label: String, recall: Double, precision: Double, f1Score: Double)] = []
        
        for label in labels {
            var truePositives = 0
            var falsePositives = 0
            var falseNegatives = 0
            
            if let labelIndex = labels.firstIndex(of: label) {
                truePositives = confusionMatrix[labelIndex][labelIndex]
                
                // Calculate false positives (sum of column except true positive)
                for row in confusionMatrix {
                    falsePositives += row[labelIndex]
                }
                falsePositives -= truePositives
                
                // Calculate false negatives (sum of row except true positive)
                falseNegatives = confusionMatrix[labelIndex].reduce(0, +) - truePositives
            }
            
            let recall = (truePositives + falseNegatives) > 0 ? Double(truePositives) / Double(truePositives + falseNegatives) : 0.0
            let precision = (truePositives + falsePositives) > 0 ? Double(truePositives) / Double(truePositives + falsePositives) : 0.0
            let f1Score = (precision + recall) > 0 ? 2 * (precision * recall) / (precision + recall) : 0.0
            
            metrics.append((label: label, recall: recall, precision: precision, f1Score: f1Score))
        }
        
        self.detailedMetrics = metrics
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
        
        // Print detailed metrics for each class
        for metric in detailedMetrics {
            print(String(format: "  %@: 再現率 %.1f%%, 適合率 %.1f%%, F1スコア %.1f%%",
                metric.label,
                metric.recall * 100,
                metric.precision * 100,
                metric.f1Score * 100))
        }
    }
    
    public func calculateMetrics() -> [(label: String, recall: Double, precision: Double, f1Score: Double)] {
        return detailedMetrics
    }
} 

import XCTest
@testable import CSConfusionMatrix

final class CSMultiClassConfusionMatrixTests: XCTestCase {
    func testMetrics() {
        // 猫、犬、鳥の3クラス分類の例
        // 各クラス100サンプルずつ
        let matrix = CSMultiClassConfusionMatrix(
            matrix: [
                [80, 10, 10],  // 猫の行：実際が猫で、予測が[猫,犬,鳥]の数
                [10, 80, 10],  // 犬の行：実際が犬で、予測が[猫,犬,鳥]の数
                [10, 10, 80]   // 鳥の行：実際が鳥で、予測が[猫,犬,鳥]の数
            ],
            labels: ["猫", "犬", "鳥"]
        )
        
        let metrics = matrix.calculateMetrics()
        
        // 猫の性能指標
        // 再現率 = 80 / 100 = 0.8 (80%)
        // 適合率 = 80 / 100 = 0.8 (80%)
        XCTAssertEqual(metrics[0].label, "猫")
        XCTAssertEqual(metrics[0].recall, 0.8, accuracy: 0.001)
        XCTAssertEqual(metrics[0].precision, 0.8, accuracy: 0.001)
        
        // 犬の性能指標
        // 再現率 = 80 / 100 = 0.8 (80%)
        // 適合率 = 80 / 100 = 0.8 (80%)
        XCTAssertEqual(metrics[1].label, "犬")
        XCTAssertEqual(metrics[1].recall, 0.8, accuracy: 0.001)
        XCTAssertEqual(metrics[1].precision, 0.8, accuracy: 0.001)
        
        // 鳥の性能指標
        // 再現率 = 80 / 100 = 0.8 (80%)
        // 適合率 = 80 / 100 = 0.8 (80%)
        XCTAssertEqual(metrics[2].label, "鳥")
        XCTAssertEqual(metrics[2].recall, 0.8, accuracy: 0.001)
        XCTAssertEqual(metrics[2].precision, 0.8, accuracy: 0.001)
    }
    
    func testZeroDivision() {
        // 全て0の場合
        let matrix = CSMultiClassConfusionMatrix(
            matrix: [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ],
            labels: ["猫", "犬", "鳥"]
        )
        
        let metrics = matrix.calculateMetrics()
        
        for metric in metrics {
            XCTAssertEqual(metric.recall, 0.0)
            XCTAssertEqual(metric.precision, 0.0)
        }
    }
    
    func testPerfectScore() {
        // 全て正解の場合（100%）
        let matrix = CSMultiClassConfusionMatrix(
            matrix: [
                [100, 0, 0],
                [0, 100, 0],
                [0, 0, 100]
            ],
            labels: ["猫", "犬", "鳥"]
        )
        
        let metrics = matrix.calculateMetrics()
        
        for metric in metrics {
            XCTAssertEqual(metric.recall, 1.0)
            XCTAssertEqual(metric.precision, 1.0)
        }
    }
    
    func testHalfCorrect() {
        // 50%正解の場合
        let matrix = CSMultiClassConfusionMatrix(
            matrix: [
                [50, 25, 25],  // 猫の行：実際が猫で、予測が[猫,犬,鳥]の数
                [25, 50, 25],  // 犬の行：実際が犬で、予測が[猫,犬,鳥]の数
                [25, 25, 50]   // 鳥の行：実際が鳥で、予測が[猫,犬,鳥]の数
            ],
            labels: ["猫", "犬", "鳥"]
        )
        
        let metrics = matrix.calculateMetrics()
        
        for metric in metrics {
            XCTAssertEqual(metric.recall, 0.5, accuracy: 0.001)
            XCTAssertEqual(metric.precision, 0.5, accuracy: 0.001)
        }
    }
} 
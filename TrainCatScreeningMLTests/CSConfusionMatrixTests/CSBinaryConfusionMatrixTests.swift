import XCTest
@testable import CSConfusionMatrix

final class CSBinaryConfusionMatrixTests: XCTestCase {
    func testMetrics() {
        // 猫vs犬の判定の例（各クラス100サンプル）
        // TP: 実際が猫で予測も猫 = 80
        // FP: 実際は犬だが予測が猫 = 20
        // FN: 実際は猫だが予測が犬 = 20
        // TN: 実際が犬で予測も犬 = 80
        let matrix = CSBinaryConfusionMatrix(
            truePositive: 80,
            falsePositive: 20,
            falseNegative: 20,
            trueNegative: 80
        )
        
        // 再現率 = TP / (TP + FN) = 80 / 100 = 0.8 (80%)
        XCTAssertEqual(matrix.recall, 0.8, accuracy: 0.001)
        
        // 適合率 = TP / (TP + FP) = 80 / 100 = 0.8 (80%)
        XCTAssertEqual(matrix.precision, 0.8, accuracy: 0.001)
        
        // 精度 = (TP + TN) / 全サンプル = (80 + 80) / 200 = 0.8 (80%)
        XCTAssertEqual(matrix.accuracy, 0.8, accuracy: 0.001)
        
        // F1スコア = 2 * (precision * recall) / (precision + recall)
        // = 2 * (0.8 * 0.8) / (0.8 + 0.8) = 0.8 (80%)
        XCTAssertEqual(matrix.f1Score, 0.8, accuracy: 0.001)
    }
    
    func testZeroDivision() {
        // 全て0の場合（0%）
        let matrix = CSBinaryConfusionMatrix(
            truePositive: 0,
            falsePositive: 0,
            falseNegative: 0,
            trueNegative: 0
        )
        
        XCTAssertEqual(matrix.recall, 0.0)
        XCTAssertEqual(matrix.precision, 0.0)
        XCTAssertEqual(matrix.accuracy, 0.0)
        XCTAssertEqual(matrix.f1Score, 0.0)
    }
    
    func testPerfectScore() {
        // 全て正解の場合（100%）
        let matrix = CSBinaryConfusionMatrix(
            truePositive: 100,
            falsePositive: 0,
            falseNegative: 0,
            trueNegative: 100
        )
        
        XCTAssertEqual(matrix.recall, 1.0)
        XCTAssertEqual(matrix.precision, 1.0)
        XCTAssertEqual(matrix.accuracy, 1.0)
        XCTAssertEqual(matrix.f1Score, 1.0)
    }
    
    func testHalfCorrect() {
        // 50%正解の場合
        let matrix = CSBinaryConfusionMatrix(
            truePositive: 50,
            falsePositive: 50,
            falseNegative: 50,
            trueNegative: 50
        )
        
        // 全ての指標が0.5（50%）になる
        XCTAssertEqual(matrix.recall, 0.5, accuracy: 0.001)
        XCTAssertEqual(matrix.precision, 0.5, accuracy: 0.001)
        XCTAssertEqual(matrix.accuracy, 0.5, accuracy: 0.001)
        XCTAssertEqual(matrix.f1Score, 0.5, accuracy: 0.001)
    }
} 
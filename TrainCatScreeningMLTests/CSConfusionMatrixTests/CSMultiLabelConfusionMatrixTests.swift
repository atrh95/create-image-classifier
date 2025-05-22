import XCTest
@testable import CSConfusionMatrix

final class CSMultiLabelConfusionMatrixTests: XCTestCase {
    func testPerfectScore() {
        // 全て正解の場合（100%）
        let labels = ["犬", "猫", "馬"]
        let predictions: [(trueLabels: Set<String>, predictedLabels: Set<String>)] = [
            (Set(["犬"]), Set(["犬"])),
            (Set(["猫"]), Set(["猫"])),
            (Set(["馬"]), Set(["馬"]))
        ]
        
        let matrix = CSMultiLabelConfusionMatrix(
            predictions: predictions,
            labels: labels
        )
        
        let metrics = matrix.calculateMetrics()
        
        for metric in metrics {
            XCTAssertNotNil(metric.recall)
            XCTAssertNotNil(metric.precision)
            XCTAssertNotNil(metric.f1Score)
            XCTAssertEqual(metric.recall!, 1.0)
            XCTAssertEqual(metric.precision!, 1.0)
            XCTAssertEqual(metric.f1Score!, 1.0)
        }
    }
    
    func testHalfCorrect() {
        // 50%正解の場合
        let labels = ["犬", "猫", "馬"]
        let predictions: [(trueLabels: Set<String>, predictedLabels: Set<String>)] = [
            (Set(["犬"]), Set(["犬"])), // 正解
            (Set(["猫"]), Set([])), // 見落とし
            (Set(["馬"]), Set(["馬", "犬"])) // 過剰予測
        ]
        
        let matrix = CSMultiLabelConfusionMatrix(
            predictions: predictions,
            labels: labels
        )
        
        let metrics = matrix.calculateMetrics()
        
        // 犬: 1/1 TP, 1 FP
        let dog = metrics.first { $0.label == "犬" }!
        XCTAssertNotNil(dog.recall)
        XCTAssertNotNil(dog.precision)
        XCTAssertNotNil(dog.f1Score)
        XCTAssertEqual(dog.recall!, 1.0)
        XCTAssertEqual(dog.precision!, 0.5)
        XCTAssertEqual(dog.f1Score!, 2/3, accuracy: 0.001)
        
        // 猫: 0/1 TP, 0 FP
        let cat = metrics.first { $0.label == "猫" }!
        XCTAssertEqual(cat.recall, 0.0)  // TP = 0, FN = 1 なので再現率は0.0
        XCTAssertNil(cat.precision)       // TP = 0, FP = 0 なので適合率は計算不能
        XCTAssertNil(cat.f1Score)         // precision が nil なので F1スコアも計算不能
        
        // 馬: 1/1 TP, 0 FP
        let horse = metrics.first { $0.label == "馬" }!
        XCTAssertNotNil(horse.recall)
        XCTAssertNotNil(horse.precision)
        XCTAssertNotNil(horse.f1Score)
        XCTAssertEqual(horse.recall!, 1.0)
        XCTAssertEqual(horse.precision!, 1.0)
        XCTAssertEqual(horse.f1Score!, 1.0)
    }
} 
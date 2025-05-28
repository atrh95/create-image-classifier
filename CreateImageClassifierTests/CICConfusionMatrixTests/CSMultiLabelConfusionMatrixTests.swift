@testable import CICConfusionMatrix
import XCTest

final class CSMultiLabelConfusionMatrixTests: XCTestCase {
    func testPerfectScore() {
        // 全て正解の場合（100%）
        let labels = ["犬", "猫", "馬"]
        let predictions: [(trueLabels: Set<String>, predictedLabels: Set<String>)] = [
            (Set(["犬"]), Set(["犬"])),
            (Set(["猫"]), Set(["猫"])),
            (Set(["馬"]), Set(["馬"])),
        ]

        let matrix = CSMultiLabelConfusionMatrix(
            predictions: predictions,
            labels: labels
        )

        let metrics = matrix.calculateMetrics()

        for metric in metrics {
            guard let recall = metric.recall,
                  let precision = metric.precision,
                  let f1Score = metric.f1Score else {
                XCTFail("メトリクスの計算に失敗しました")
                return
            }
            XCTAssertEqual(recall, 1.0)
            XCTAssertEqual(precision, 1.0)
            XCTAssertEqual(f1Score, 1.0)
        }
    }

    func testHalfCorrect() {
        // 50%正解の場合
        let labels = ["犬", "猫", "馬"]
        let predictions: [(trueLabels: Set<String>, predictedLabels: Set<String>)] = [
            (Set(["犬"]), Set(["犬"])), // 正解
            (Set(["猫"]), Set([])), // 見落とし
            (Set(["馬"]), Set(["馬", "犬"])), // 過剰予測
        ]

        let matrix = CSMultiLabelConfusionMatrix(
            predictions: predictions,
            labels: labels
        )

        let metrics = matrix.calculateMetrics()

        // 犬: 1/1 TP, 1 FP
        guard let dog = metrics.first(where: { $0.label == "犬" }),
              let dogRecall = dog.recall,
              let dogPrecision = dog.precision,
              let dogF1Score = dog.f1Score else {
            XCTFail("犬のメトリクスが見つからないか、計算に失敗しました")
            return
        }
        XCTAssertEqual(dogRecall, 1.0)
        XCTAssertEqual(dogPrecision, 0.5)
        XCTAssertEqual(dogF1Score, 2 / 3, accuracy: 0.001)

        // 猫: 0/1 TP, 0 FP
        guard let cat = metrics.first(where: { $0.label == "猫" }) else {
            XCTFail("猫のメトリクスが見つかりません")
            return
        }
        XCTAssertEqual(cat.recall, 0.0) // TP = 0, FN = 1 なので再現率は0.0
        XCTAssertNil(cat.precision) // TP = 0, FP = 0 なので適合率は計算不可
        XCTAssertNil(cat.f1Score) // precision が nil なので F1スコアも計算不可

        // 馬: 1/1 TP, 0 FP
        guard let horse = metrics.first(where: { $0.label == "馬" }),
              let horseRecall = horse.recall,
              let horsePrecision = horse.precision,
              let horseF1Score = horse.f1Score else {
            XCTFail("馬のメトリクスが見つからないか、計算に失敗しました")
            return
        }
        XCTAssertEqual(horseRecall, 1.0)
        XCTAssertEqual(horsePrecision, 1.0)
        XCTAssertEqual(horseF1Score, 1.0)
    }
}

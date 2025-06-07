import CICConfusionMatrix
import CreateML
import XCTest

final class CICMultiClassConfusionMatrixTests: XCTestCase {
    private func createMultiClassDataTable(
        matrix: [[Int]],
        labels: [String]
    ) throws -> MLDataTable {
        var predictedValues: [String] = []
        var actualValues: [String] = []
        var countValues: [Int] = []

        for (actualIndex, actualLabel) in labels.enumerated() {
            for (predictedIndex, predictedLabel) in labels.enumerated() where matrix[actualIndex][predictedIndex] > 0 {
                predictedValues.append(predictedLabel)
                actualValues.append(actualLabel)
                countValues.append(matrix[actualIndex][predictedIndex])
            }
        }

        return try MLDataTable(dictionary: [
            "Predicted": predictedValues,
            "True Label": actualValues,
            "Count": countValues,
        ])
    }

    func testValidateDataTable() throws {
        // 正常なデータテーブル
        let matrix = [
            [80, 10, 10],
            [10, 80, 10],
            [10, 10, 80],
        ]
        let labels = ["猫", "犬", "鳥"]

        let validDataTable = try createMultiClassDataTable(matrix: matrix, labels: labels)
        XCTAssertTrue(
            CICMultiClassConfusionMatrix.validateDataTable(
                validDataTable,
                predictedColumn: "Predicted",
                actualColumn: "True Label"
            ),
            "正常なデータテーブルは検証に成功するはずです"
        )

        // 空のデータテーブル
        let emptyDataTable = try MLDataTable(dictionary: [
            "Predicted": [String](),
            "True Label": [String](),
            "Count": [Int](),
        ])

        XCTAssertFalse(
            CICMultiClassConfusionMatrix.validateDataTable(
                emptyDataTable,
                predictedColumn: "Predicted",
                actualColumn: "True Label"
            ),
            "空のデータテーブルは検証に失敗するはずです"
        )

        // 必要な列が存在しないデータテーブル
        let missingColumnDataTable = try MLDataTable(dictionary: [
            "wrong_column": ["猫", "犬", "鳥"],
        ])

        XCTAssertFalse(
            CICMultiClassConfusionMatrix.validateDataTable(
                missingColumnDataTable,
                predictedColumn: "Predicted",
                actualColumn: "True Label"
            ),
            "必要な列が存在しないデータテーブルは検証に失敗するはずです"
        )

        // ラベルが存在しないデータテーブル
        let noLabelDataTable = try MLDataTable(dictionary: [
            "Predicted": [String](),
            "True Label": [String](),
            "Count": [Int](),
        ])

        XCTAssertFalse(
            CICMultiClassConfusionMatrix.validateDataTable(
                noLabelDataTable,
                predictedColumn: "Predicted",
                actualColumn: "True Label"
            ),
            "ラベルが存在しないデータテーブルは検証に失敗するはずです"
        )

        // 予測値と実際の値が一致しないデータテーブル
        let predictedValues = ["猫", "犬", "鳥"]
        let actualValues = ["猫", "犬", "馬"] // 馬は予測値に存在しない
        let counts = [1, 1, 1]

        let mismatchedDataTable = try MLDataTable(dictionary: [
            "Predicted": predictedValues,
            "True Label": actualValues,
            "Count": counts,
        ])

        XCTAssertFalse(
            CICMultiClassConfusionMatrix.validateDataTable(
                mismatchedDataTable,
                predictedColumn: "Predicted",
                actualColumn: "True Label"
            ),
            "予測値と実際の値が一致しないデータテーブルは検証に失敗するはずです"
        )
    }

    func testMetrics() throws {
        // 猫、犬、鳥の3クラス分類の例
        // 各クラス100サンプルずつ
        let matrix = [
            [80, 10, 10], // 猫の行：実際が猫で、予測が[猫,犬,鳥]の数
            [10, 80, 10], // 犬の行：実際が犬で、予測が[猫,犬,鳥]の数
            [10, 10, 80], // 鳥の行：実際が鳥で、予測が[猫,犬,鳥]の数
        ]
        let labels = ["猫", "犬", "鳥"]

        let dataTable = try createMultiClassDataTable(matrix: matrix, labels: labels)
        guard let confusionMatrix = CICMultiClassConfusionMatrix(
            dataTable: dataTable,
            predictedColumn: "Predicted",
            actualColumn: "True Label"
        ) else {
            XCTFail("混同行列の作成に失敗しました")
            return
        }

        let metrics = confusionMatrix.calculateMetrics()

        // Print the actual order of labels
        print("Actual labels order:")
        for (index, metric) in metrics.enumerated() {
            print("\(index): \(metric.label)")
        }

        // 犬の性能指標（アルファベット順で最初）
        // 再現率 = 80 / 100 = 0.8 (80%)
        // 適合率 = 80 / 100 = 0.8 (80%)
        XCTAssertEqual(metrics[0].label, "犬")
        XCTAssertEqual(metrics[0].recall, 0.8, accuracy: 0.001)
        XCTAssertEqual(metrics[0].precision, 0.8, accuracy: 0.001)
        XCTAssertEqual(metrics[0].f1Score, 0.8, accuracy: 0.001)

        // 猫の性能指標（アルファベット順で2番目）
        // 再現率 = 80 / 100 = 0.8 (80%)
        // 適合率 = 80 / 100 = 0.8 (80%)
        // F1スコア = 2 * (0.8 * 0.8) / (0.8 + 0.8) = 0.8 (80%)
        XCTAssertEqual(metrics[1].label, "猫")
        XCTAssertEqual(metrics[1].recall, 0.8, accuracy: 0.001)
        XCTAssertEqual(metrics[1].precision, 0.8, accuracy: 0.001)
        XCTAssertEqual(metrics[1].f1Score, 0.8, accuracy: 0.001)

        // 鳥の性能指標（アルファベット順で3番目）
        // 再現率 = 80 / 100 = 0.8 (80%)
        // 適合率 = 80 / 100 = 0.8 (80%)
        // F1スコア = 2 * (0.8 * 0.8) / (0.8 + 0.8) = 0.8 (80%)
        XCTAssertEqual(metrics[2].label, "鳥")
        XCTAssertEqual(metrics[2].recall, 0.8, accuracy: 0.001)
        XCTAssertEqual(metrics[2].precision, 0.8, accuracy: 0.001)
        XCTAssertEqual(metrics[2].f1Score, 0.8, accuracy: 0.001)
    }

    func testPerfectScore() throws {
        // 全て正解の場合（100%）
        let matrix = [
            [100, 0, 0],
            [0, 100, 0],
            [0, 0, 100],
        ]
        let labels = ["猫", "犬", "鳥"]

        let dataTable = try createMultiClassDataTable(matrix: matrix, labels: labels)
        guard let confusionMatrix = CICMultiClassConfusionMatrix(
            dataTable: dataTable,
            predictedColumn: "Predicted",
            actualColumn: "True Label"
        ) else {
            XCTFail("混同行列の作成に失敗しました")
            return
        }

        let metrics = confusionMatrix.calculateMetrics()

        for metric in metrics {
            XCTAssertEqual(metric.recall, 1.0)
            XCTAssertEqual(metric.precision, 1.0)
            XCTAssertEqual(metric.f1Score, 1.0)
        }
    }

    func testHalfCorrect() throws {
        // 50%正解の場合
        let matrix = [
            [50, 25, 25], // 猫の行：実際が猫で、予測が[猫,犬,鳥]の数
            [25, 50, 25], // 犬の行：実際が犬で、予測が[猫,犬,鳥]の数
            [25, 25, 50], // 鳥の行：実際が鳥で、予測が[猫,犬,鳥]の数
        ]
        let labels = ["猫", "犬", "鳥"]

        let dataTable = try createMultiClassDataTable(matrix: matrix, labels: labels)
        guard let confusionMatrix = CICMultiClassConfusionMatrix(
            dataTable: dataTable,
            predictedColumn: "Predicted",
            actualColumn: "True Label"
        ) else {
            XCTFail("混同行列の作成に失敗しました")
            return
        }

        let metrics = confusionMatrix.calculateMetrics()

        for metric in metrics {
            XCTAssertEqual(metric.recall, 0.5, accuracy: 0.001)
            XCTAssertEqual(metric.precision, 0.5, accuracy: 0.001)
            XCTAssertEqual(metric.f1Score, 0.5, accuracy: 0.001)
        }
    }
}

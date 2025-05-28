import CreateML
@testable import CICConfusionMatrix
import XCTest

final class CSBinaryConfusionMatrixTests: XCTestCase {
    private func createBinaryDataTable(
        truePositive: Int,
        falsePositive: Int,
        falseNegative: Int,
        trueNegative: Int
    ) throws -> MLDataTable {
        var predictedValues: [String] = []
        var actualValues: [String] = []
        var countValues: [Int] = []

        // Add TP samples
        predictedValues.append("猫")
        actualValues.append("猫")
        countValues.append(truePositive)

        // Add FP samples
        predictedValues.append("猫")
        actualValues.append("犬")
        countValues.append(falsePositive)

        // Add FN samples
        predictedValues.append("犬")
        actualValues.append("猫")
        countValues.append(falseNegative)

        // Add TN samples
        predictedValues.append("犬")
        actualValues.append("犬")
        countValues.append(trueNegative)

        return try MLDataTable(dictionary: [
            "Predicted": predictedValues,
            "True Label": actualValues,
            "Count": countValues,
        ])
    }

    func testValidateDataTable() throws {
        // 正常なデータテーブル
        let validDataTable = try createBinaryDataTable(
            truePositive: 1,
            falsePositive: 1,
            falseNegative: 1,
            trueNegative: 1
        )

        XCTAssertTrue(
            CSBinaryConfusionMatrix.validateDataTable(
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
            CSBinaryConfusionMatrix.validateDataTable(
                emptyDataTable,
                predictedColumn: "Predicted",
                actualColumn: "True Label"
            ),
            "空のデータテーブルは検証に失敗するはずです"
        )

        // 必要な列が存在しないデータテーブル
        let missingColumnDataTable = try MLDataTable(dictionary: [
            "wrong_column": ["猫", "犬"],
        ])

        XCTAssertFalse(
            CSBinaryConfusionMatrix.validateDataTable(
                missingColumnDataTable,
                predictedColumn: "Predicted",
                actualColumn: "True Label"
            ),
            "必要な列が存在しないデータテーブルは検証に失敗するはずです"
        )

        // 1クラスのみのデータテーブル
        let singleClassPredictedValues = ["猫", "猫"]
        let singleClassActualValues = ["猫", "猫"]
        let singleClassCounts = [1, 1]

        let singleClassDataTable = try MLDataTable(dictionary: [
            "Predicted": singleClassPredictedValues,
            "True Label": singleClassActualValues,
            "Count": singleClassCounts,
        ])

        XCTAssertFalse(
            CSBinaryConfusionMatrix.validateDataTable(
                singleClassDataTable,
                predictedColumn: "Predicted",
                actualColumn: "True Label"
            ),
            "1クラスのみのデータテーブルは検証に失敗するはずです"
        )

        // 3クラスのデータテーブル
        let threeClassPredictedValues = ["猫", "犬", "鳥"]
        let threeClassActualValues = ["猫", "犬", "鳥"]
        let threeClassCounts = [1, 1, 1]

        let threeClassDataTable = try MLDataTable(dictionary: [
            "Predicted": threeClassPredictedValues,
            "True Label": threeClassActualValues,
            "Count": threeClassCounts,
        ])

        XCTAssertFalse(
            CSBinaryConfusionMatrix.validateDataTable(
                threeClassDataTable,
                predictedColumn: "Predicted",
                actualColumn: "True Label"
            ),
            "3クラスのデータテーブルは検証に失敗するはずです"
        )
    }

    func testMetrics() throws {
        // 猫vs犬の判定の例（各クラス100サンプル）
        // TP: 実際が猫で予測も猫 = 80
        // FP: 実際は犬だが予測が猫 = 20
        // FN: 実際は猫だが予測が犬 = 20
        // TN: 実際が犬で予測も犬 = 80
        let dataTable = try createBinaryDataTable(
            truePositive: 80,
            falsePositive: 20,
            falseNegative: 20,
            trueNegative: 80
        )

        guard let matrix = CSBinaryConfusionMatrix(
            dataTable: dataTable,
            predictedColumn: "Predicted",
            actualColumn: "True Label"
        ) else {
            XCTFail("混同行列の作成に失敗しました")
            return
        }

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

    func testPerfectScore() throws {
        // 全て正解の場合（100%）
        let dataTable = try createBinaryDataTable(
            truePositive: 100,
            falsePositive: 0,
            falseNegative: 0,
            trueNegative: 100
        )

        guard let matrix = CSBinaryConfusionMatrix(
            dataTable: dataTable,
            predictedColumn: "Predicted",
            actualColumn: "True Label"
        ) else {
            XCTFail("混同行列の作成に失敗しました")
            return
        }

        XCTAssertEqual(matrix.recall, 1.0)
        XCTAssertEqual(matrix.precision, 1.0)
        XCTAssertEqual(matrix.accuracy, 1.0)
        XCTAssertEqual(matrix.f1Score, 1.0)
    }

    func testHalfCorrect() throws {
        // 50%正解の場合
        let dataTable = try createBinaryDataTable(
            truePositive: 50,
            falsePositive: 50,
            falseNegative: 50,
            trueNegative: 50
        )

        guard let matrix = CSBinaryConfusionMatrix(
            dataTable: dataTable,
            predictedColumn: "Predicted",
            actualColumn: "True Label"
        ) else {
            XCTFail("混同行列の作成に失敗しました")
            return
        }

        // 全ての指標が0.5（50%）になる
        XCTAssertEqual(matrix.recall, 0.5, accuracy: 0.001)
        XCTAssertEqual(matrix.precision, 0.5, accuracy: 0.001)
        XCTAssertEqual(matrix.accuracy, 0.5, accuracy: 0.001)
        XCTAssertEqual(matrix.f1Score, 0.5, accuracy: 0.001)
    }
}

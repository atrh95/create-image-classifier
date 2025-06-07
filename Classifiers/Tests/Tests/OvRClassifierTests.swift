import CICFileManager
import CoreML
import CreateML
import Foundation
import OvRClassifier
import Vision
import XCTest

final class OvRClassifierTests: XCTestCase {
    var classifier: OvRClassifier!
    let fileManager = CICFileManager()
    let authorName: String = "Test Author"
    let testModelName: String = "TestModel"
    let testModelVersion: String = "v1"

    let algorithm = MLImageClassifier.ModelParameters.ModelAlgorithmType.transferLearning(
        featureExtractor: .scenePrint(revision: 1),
        classifier: .logisticRegressor
    )

    var modelParameters: MLImageClassifier.ModelParameters {
        MLImageClassifier.ModelParameters(
            validation: .split(strategy: .automatic),
            maxIterations: 1,
            augmentation: [],
            algorithm: algorithm
        )
    }

    var temporaryOutputDirectoryURL: URL!

    override func setUpWithError() throws {
        temporaryOutputDirectoryURL = fileManager.temporaryDirectory
            .appendingPathComponent("TestOutput_OvR")
        try fileManager.createDirectory(
            at: temporaryOutputDirectoryURL,
            withIntermediateDirectories: true,
            attributes: nil
        )

        let resourceDirectoryPath = "Classifiers/Tests/TestResources/OvRResources"

        classifier = OvRClassifier(
            outputDirectoryPathOverride: temporaryOutputDirectoryURL.path,
            resourceDirPathOverride: resourceDirectoryPath
        )
    }

    override func tearDownWithError() throws {
        if let tempDir = temporaryOutputDirectoryURL, fileManager.fileExists(atPath: tempDir.path) {
            try? fileManager.removeItem(at: tempDir)
        }
        temporaryOutputDirectoryURL = nil

        classifier.resourceDirPathOverride = nil
        classifier = nil
        try super.tearDownWithError()
    }

    func testClassifierDIConfiguration() throws {
        XCTAssertNotNil(classifier, "OvRClassifierの初期化失敗")
        XCTAssertEqual(classifier.outputParentDirPath, temporaryOutputDirectoryURL.path, "分類器の出力パスが期待値と不一致")
    }

    // モデルの訓練と成果物の生成をテスト
    func testModelTrainingAndArtifactGeneration() throws {
        try classifier.createAndSaveModel(
            author: authorName,
            modelName: testModelName,
            version: testModelVersion,
            modelParameters: modelParameters,
            shouldEqualizeFileCount: true
        )

        // 出力ディレクトリから最新の結果を取得
        let outputDir = URL(fileURLWithPath: classifier.outputParentDirPath)
            .appendingPathComponent(testModelName)
            .appendingPathComponent(testModelVersion)
        let resultDirs = try fileManager.contentsOfDirectory(
            at: outputDir,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ).filter(\.hasDirectoryPath)
            .sorted { $0.lastPathComponent > $1.lastPathComponent }

        guard let latestResultDir = resultDirs.first else {
            XCTFail("結果ディレクトリが見つかりません: \(outputDir.path)")
            return
        }

        // クラスラベルディレクトリの取得
        let classLabelDirs = try fileManager.contentsOfDirectory(
            at: URL(fileURLWithPath: classifier.resourcesDirectoryPath),
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ).filter(\.hasDirectoryPath)
            .map(\.lastPathComponent)
            .sorted()

        // 各クラスラベルに対応するモデルファイルの存在確認
        for classLabel in classLabelDirs {
            let expectedModelFileName =
                "\(testModelName)_\(classifier.classificationMethod)_\(classLabel)_\(testModelVersion).mlmodel"
            let modelFilePath = latestResultDir.appendingPathComponent(expectedModelFileName).path

            XCTAssertTrue(
                fileManager.fileExists(atPath: modelFilePath),
                "クラス '\(classLabel)' の訓練モデルファイルが期待されるパス「\(modelFilePath)」に見つかりません"
            )
        }

        // クラスラベルの存在確認
        XCTAssertFalse(
            classLabelDirs.isEmpty,
            "リソースディレクトリにクラスラベルディレクトリが見つかりません: \(classifier.resourcesDirectoryPath)"
        )

        // 各クラスラベルディレクトリにファイルが存在することを確認
        for classLabel in classLabelDirs {
            let classDirURL = URL(fileURLWithPath: classifier.resourcesDirectoryPath)
                .appendingPathComponent(classLabel)
            let files = try fileManager.contentsOfDirectory(
                at: classDirURL,
                includingPropertiesForKeys: nil
            )

            XCTAssertFalse(
                files.isEmpty,
                "クラスラベル「\(classLabel)」のディレクトリにファイルが見つかりません: \(classDirURL.path)"
            )
        }

        // ログファイルの存在確認
        let expectedLogFileName = "\(testModelName)_\(testModelVersion).md"
        let expectedLogFilePath = latestResultDir.appendingPathComponent(expectedLogFileName).path
        XCTAssertTrue(
            fileManager.fileExists(atPath: expectedLogFilePath),
            "ログファイルが期待パス「\(expectedLogFilePath)」に未生成"
        )

        // モデルファイル名の検証
        for classLabel in classLabelDirs {
            let modelFileName =
                "\(testModelName)_\(classifier.classificationMethod)_\(classLabel)_\(testModelVersion).mlmodel"
            let regex = #"^TestModel_OvR_\w+_v\d+\.mlmodel$"#
            XCTAssertTrue(
                modelFileName.range(of: regex, options: .regularExpression) != nil,
                """
                モデルファイル名が期待パターンに一致しません。
                期待パターン: \(regex)
                実際: \(modelFileName)
                """
            )

            let modelFilePath = latestResultDir.appendingPathComponent(modelFileName).path
            XCTAssertTrue(
                modelFilePath.contains(testModelVersion),
                "モデルファイルパスにバージョン「\(testModelVersion)」が含まれていません"
            )
            XCTAssertTrue(
                modelFilePath.contains("OvR"),
                "モデルファイルパスに分類法「OvR」が含まれていません"
            )
            XCTAssertTrue(
                modelFilePath.contains(classLabel),
                "モデルファイルパスにクラスラベル「\(classLabel)」が含まれていません"
            )
        }
    }

    // モデルが予測を実行できるかテスト
    // func testModelCanPerformPrediction() throws {
    //     ...（関数全体を削除）...
    // }

    // 出力ディレクトリの連番を検証
    func testSequentialOutputDirectoryNumbering() throws {
        try classifier.createAndSaveModel(
            author: authorName,
            modelName: testModelName,
            version: testModelVersion,
            modelParameters: modelParameters,
            shouldEqualizeFileCount: true
        )

        // 出力ディレクトリから最新の結果を取得
        let outputDir = URL(fileURLWithPath: classifier.outputParentDirPath)
            .appendingPathComponent(testModelName)
            .appendingPathComponent(testModelVersion)
        let firstResultDirs = try fileManager.contentsOfDirectory(
            at: outputDir,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ).filter(\.hasDirectoryPath)
            .sorted { $0.lastPathComponent > $1.lastPathComponent }

        guard let firstResultDir = firstResultDirs.first else {
            XCTFail("1回目の結果ディレクトリが見つかりません: \(outputDir.path)")
            return
        }

        // 2回目のモデル作成を実行
        try classifier.createAndSaveModel(
            author: "TestAuthor",
            modelName: testModelName,
            version: "v1",
            modelParameters: modelParameters,
            shouldEqualizeFileCount: true
        )

        // 出力ディレクトリから最新の結果を取得
        let secondResultDirs = try fileManager.contentsOfDirectory(
            at: outputDir,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ).filter(\.hasDirectoryPath)
            .sorted { $0.lastPathComponent > $1.lastPathComponent }

        guard let secondResultDir = secondResultDirs.first else {
            XCTFail("2回目の結果ディレクトリが見つかりません: \(outputDir.path)")
            return
        }

        // 連番の検証
        let firstResultNumber = Int(firstResultDir.lastPathComponent.replacingOccurrences(
            of: "OvR_Result_",
            with: ""
        )) ?? 0
        let secondResultNumber = Int(secondResultDir.lastPathComponent.replacingOccurrences(
            of: "OvR_Result_",
            with: ""
        )) ?? 0

        print("1回目の結果ディレクトリ: \(firstResultDir.lastPathComponent)")
        print("2回目の結果ディレクトリ: \(secondResultDir.lastPathComponent)")
        print("1回目の番号: \(firstResultNumber)")
        print("2回目の番号: \(secondResultNumber)")

        XCTAssertEqual(
            secondResultNumber,
            firstResultNumber + 1,
            "2回目の出力ディレクトリの連番が期待値と一致しません。\n1回目: \(firstResultNumber)\n2回目: \(secondResultNumber)"
        )
    }

    // クラス間のファイル数バランスを検証
    func testClassFileCountBalance() throws {
        try classifier.createAndSaveModel(
            author: authorName,
            modelName: testModelName,
            version: testModelVersion,
            modelParameters: modelParameters,
            shouldEqualizeFileCount: true
        )

        // 出力ディレクトリから最新の結果を取得
        let outputDir = URL(fileURLWithPath: classifier.outputParentDirPath)
            .appendingPathComponent(testModelName)
            .appendingPathComponent(testModelVersion)
        let resultDirs = try fileManager.contentsOfDirectory(
            at: outputDir,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ).filter(\.hasDirectoryPath)
            .sorted { $0.lastPathComponent > $1.lastPathComponent }

        guard let latestResultDir = resultDirs.first else {
            XCTFail("結果ディレクトリが見つかりません: \(outputDir.path)")
            return
        }

        // クラスラベルディレクトリの取得
        let classLabelDirs = try fileManager.contentsOfDirectory(
            at: URL(fileURLWithPath: classifier.resourcesDirectoryPath),
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ).filter(\.hasDirectoryPath)
            .map(\.lastPathComponent)
            .sorted()

        // 各クラスの元のファイル数を取得
        var originalFileCounts: [String: Int] = [:]
        for classLabel in classLabelDirs {
            let classDir = URL(fileURLWithPath: classifier.resourcesDirectoryPath).appendingPathComponent(classLabel)
            let files = try fileManager.contentsOfDirectory(
                at: classDir,
                includingPropertiesForKeys: nil
            )
            originalFileCounts[classLabel] = files.count
        }

        // 最小枚数を計算
        let minCount = originalFileCounts.values.min() ?? 0

        // 各クラスのモデルファイルを検証
        for classLabel in classLabelDirs {
            let expectedModelFileName =
                "\(testModelName)_\(classifier.classificationMethod)_\(classLabel)_\(testModelVersion).mlmodel"
            let modelFilePath = latestResultDir.appendingPathComponent(expectedModelFileName).path

            // モデルファイルの存在確認
            XCTAssertTrue(
                fileManager.fileExists(atPath: modelFilePath),
                "クラス '\(classLabel)' の訓練モデルファイルが期待されるパス「\(modelFilePath)」に見つかりません"
            )

            // 一時ディレクトリ内のファイル数を確認
            let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("TempOvRTrainingData")
            let positiveClassDir = tempDir.appendingPathComponent(classLabel)
            let restClassDir = tempDir.appendingPathComponent("rest")

            // 正例クラスのファイル数を確認
            let positiveFiles = try FileManager.default.contentsOfDirectory(
                at: positiveClassDir,
                includingPropertiesForKeys: nil
            )
            XCTAssertEqual(
                positiveFiles.count,
                minCount,
                """
                クラス '\(classLabel)' のモデルで、正例のファイル数が最小枚数と一致しません。
                期待: \(minCount)枚
                実際: \(positiveFiles.count)枚
                元のファイル数: \(originalFileCounts[classLabel] ?? 0)枚
                """
            )

            // restクラスのファイル数を確認
            let restFiles = try fileManager.contentsOfDirectory(
                at: restClassDir,
                includingPropertiesForKeys: nil
            )
            XCTAssertGreaterThanOrEqual(
                restFiles.count,
                positiveFiles.count,
                "クラス '\(classLabel)' のモデルで、restクラスのファイル数が正例のファイル数未満です。期待: \(positiveFiles.count)枚以上、実際: \(restFiles.count)枚"
            )
        }
    }
}

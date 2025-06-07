import CICFileManager
import CoreML
import CreateML
import Foundation
import MultiClassClassifier
import XCTest

final class MultiClassClassifierTests: XCTestCase {
    var classifier: MultiClassClassifier!
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
            .appendingPathComponent("TestOutput_MultiClass")
        try fileManager.createDirectory(
            at: temporaryOutputDirectoryURL,
            withIntermediateDirectories: true,
            attributes: nil
        )

        let resourceDirectoryPath = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // Tests/Tests
            .deletingLastPathComponent() // Tests
            .appendingPathComponent("TestResources")
            .appendingPathComponent("MultiClassResources")
            .path

        classifier = MultiClassClassifier(
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
        try classifier.createAndSaveModel(
            author: authorName,
            modelName: testModelName,
            version: testModelVersion,
            modelParameters: modelParameters,
            shouldEqualizeFileCount: true
        )

        XCTAssertNotNil(classifier, "MultiClassClassifierの初期化失敗")
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

        // モデルファイルの存在確認
        let expectedModelFileName = "\(testModelName)_\(classifier.classificationMethod)_\(testModelVersion).mlmodel"
        let modelFilePath = latestResultDir.appendingPathComponent(expectedModelFileName).path

        XCTAssertTrue(
            fileManager.fileExists(atPath: modelFilePath),
            "訓練モデルファイルが期待されるパス「\(modelFilePath)」に見つかりません"
        )

        // クラスラベルディレクトリの取得
        let classLabelDirs = try fileManager.contentsOfDirectory(
            at: URL(fileURLWithPath: classifier.resourcesDirectoryPath),
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ).filter(\.hasDirectoryPath)
            .map(\.lastPathComponent)
            .sorted()

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
        let modelFileName = expectedModelFileName
        let regex = #"^TestModel_MultiClass_v\d+\.mlmodel$"#
        XCTAssertTrue(
            modelFileName.range(of: regex, options: .regularExpression) != nil,
            """
            モデルファイル名が期待パターンに一致しません。
            期待パターン: \(regex)
            実際: \(modelFileName)
            """
        )

        XCTAssertTrue(
            modelFilePath.contains(testModelVersion),
            "モデルファイルパスにバージョン「\(testModelVersion)」が含まれていません"
        )
        XCTAssertTrue(
            modelFilePath.contains("MultiClass"),
            "モデルファイルパスに分類法「MultiClass」が含まれていません"
        )
    }

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
            of: "MultiClass_Result_",
            with: ""
        )) ?? 0
        let secondResultNumber = Int(secondResultDir.lastPathComponent.replacingOccurrences(
            of: "MultiClass_Result_",
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

        // 一時ディレクトリから各クラスのファイル数を直接確認
        let tempDir = fileManager.temporaryDirectory.appendingPathComponent("TempMultiClassTrainingData")
        for classLabel in classLabelDirs {
            let tempClassDir = tempDir.appendingPathComponent(classLabel)
            guard fileManager.fileExists(atPath: tempClassDir.path) else {
                XCTFail("クラス '\(classLabel)' の一時ディレクトリが見つかりません")
                continue
            }
            let tempFiles = try fileManager.contentsOfDirectory(at: tempClassDir, includingPropertiesForKeys: nil)
            XCTAssertEqual(
                tempFiles.count,
                minCount,
                """
                クラス '\(classLabel)' のファイル数が最小枚数と一致しません。
                期待: \(minCount)枚
                実際: \(tempFiles.count)枚
                元のファイル数: \(originalFileCounts[classLabel] ?? 0)枚
                """
            )
        }
    }
}

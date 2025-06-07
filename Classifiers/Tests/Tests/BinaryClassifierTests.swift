import BinaryClassifier
import CICFileManager
import CoreML
import CreateML
import Foundation
import XCTest

final class BinaryClassifierTests: XCTestCase {
    var classifier: BinaryClassifier!
    let fileManager = CICFileManager()
    var authorName: String = "Test Author"
    var testModelName: String = "TestModel"
    var testModelVersion: String = "v1"

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
            .appendingPathComponent("TestOutput_Binary")
        try fileManager.createDirectory(
            at: temporaryOutputDirectoryURL,
            withIntermediateDirectories: true,
            attributes: nil
        )

        let resourceDirectoryPath = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // Tests/Tests
            .deletingLastPathComponent() // Tests
            .appendingPathComponent("TestResources")
            .appendingPathComponent("BinaryResources")
            .path

        classifier = BinaryClassifier(
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
        // モデルの作成
        try classifier.createAndSaveModel(
            author: authorName,
            modelName: testModelName,
            version: testModelVersion,
            modelParameters: modelParameters,
            shouldEqualizeFileCount: true
        )

        XCTAssertNotNil(classifier, "BinaryClassifierの初期化失敗")
        XCTAssertEqual(classifier.outputParentDirPath, temporaryOutputDirectoryURL.path, "分類器の出力パスが期待値と不一致")
    }

    // モデルの訓練と成果物の生成をテスト
    func testModelTrainingAndArtifactGeneration() throws {
        // モデルの作成
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

        // Binary分類器の期待されるクラスラベル
        let classLabelDirs = try fileManager.getClassLabelDirectories(resourcesPath: classifier.resourcesDirectoryPath)
            .map(\.lastPathComponent)
            .sorted()

        // クラスラベルの存在確認
        XCTAssertFalse(
            classLabelDirs.isEmpty,
            "リソースディレクトリにクラスラベルディレクトリが見つかりません: \(classifier.resourcesDirectoryPath)"
        )

        // 各クラスラベルディレクトリにファイルが存在することを確認
        for classLabel in classLabelDirs {
            let classDirURL = URL(fileURLWithPath: classifier.resourcesDirectoryPath).appendingPathComponent(classLabel)
            let files = try fileManager.getFilesInDirectory(classDirURL)

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
        let regex = #"^TestModel_Binary_v\d+\.mlmodel$"#
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
            modelFilePath.contains("Binary"),
            "モデルファイルパスに分類法「Binary」が含まれていません"
        )
    }

    // モデルの再訓練をテスト
    func testSequentialOutputDirectoryNumbering() throws {
        // 1回目の訓練
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

        guard let firstLatestResultDir = firstResultDirs.first else {
            XCTFail("1回目の結果ディレクトリが見つかりません: \(outputDir.path)")
            return
        }

        // 2回目の訓練
        try classifier.createAndSaveModel(
            author: authorName,
            modelName: testModelName,
            version: testModelVersion,
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

        guard let secondLatestResultDir = secondResultDirs.first else {
            XCTFail("2回目の結果ディレクトリが見つかりません: \(outputDir.path)")
            return
        }

        // 2回目の結果ディレクトリが1回目より新しいことを確認
        let firstResultNumber = Int(firstLatestResultDir.lastPathComponent.replacingOccurrences(
            of: "Binary_Result_",
            with: ""
        )) ?? 0
        let secondResultNumber = Int(secondLatestResultDir.lastPathComponent.replacingOccurrences(
            of: "Binary_Result_",
            with: ""
        )) ?? 0
        XCTAssertEqual(
            secondResultNumber,
            firstResultNumber + 1,
            "2回目の出力ディレクトリの連番が期待値と一致しません。\n1回目: \(firstResultNumber)\n2回目: \(secondResultNumber)"
        )
    }

    // クラス間のファイル数バランスを検証
    func testClassFileCountBalance() throws {
        // クラスラベルディレクトリの取得
        let classLabelDirs = try fileManager.getClassLabelDirectories(resourcesPath: classifier.resourcesDirectoryPath)
            .map(\.lastPathComponent)
            .sorted()

        // 各クラスの元のファイル数を取得
        var originalFileCounts: [String: Int] = [:]
        for classLabel in classLabelDirs {
            let classDir = URL(fileURLWithPath: classifier.resourcesDirectoryPath).appendingPathComponent(classLabel)
            let files = try fileManager.getFilesInDirectory(classDir)
            originalFileCounts[classLabel] = files.count
        }

        // バランス調整された画像セットを準備
        let classDirURLs = classLabelDirs.map { classLabel in
            URL(fileURLWithPath: classifier.resourcesDirectoryPath).appendingPathComponent(classLabel)
        }
        let balancedDirs = try fileManager.prepareEqualizedMinimumImageSet(
            classDirs: classDirURLs
        )

        // 各クラスのファイル数が等しいことを確認
        var balancedFileCounts: [String: Int] = [:]
        for (className, dir) in balancedDirs {
            let files = try fileManager.getFilesInDirectory(dir)
            balancedFileCounts[className] = files.count
        }

        // 最小枚数を計算
        let minCount = originalFileCounts.values.min() ?? 0

        // 各クラスのファイル数が最小枚数に揃えられていることを確認
        for (className, count) in balancedFileCounts {
            XCTAssertEqual(
                count,
                minCount,
                """
                クラス '\(className)' のファイル数が最小枚数と一致しません。
                期待: \(minCount)枚
                実際: \(count)枚
                元のファイル数: \(originalFileCounts[className] ?? 0)枚
                """
            )
        }

        // クラス間のファイル数が等しいことを確認
        let counts = Array(balancedFileCounts.values)
        let firstCount = counts[0]
        for (index, count) in counts.enumerated() {
            XCTAssertEqual(
                count,
                firstCount,
                """
                クラス間のファイル数が等しくありません。
                クラス1 (\(Array(balancedFileCounts.keys)[0])): \(firstCount)枚
                クラス\(index + 1) (\(Array(balancedFileCounts.keys)[index])): \(count)枚
                """
            )
        }
    }
}

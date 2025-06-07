import BinaryClassifier
import CICFileManager
import CoreML
import CreateML
import Foundation
import Vision
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
            .deletingLastPathComponent() // BinaryClassifierTests
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
            modelParameters: modelParameters
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
            modelParameters: modelParameters
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

    // モデルが予測を実行できるかテスト
    func testModelCanPerformPrediction() throws {
        // モデルの作成
        try classifier.createAndSaveModel(
            author: authorName,
            modelName: testModelName,
            version: testModelVersion,
            modelParameters: modelParameters
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

        guard fileManager.fileExists(atPath: modelFilePath) else {
            XCTFail("モデルファイルが見つかりません: \(modelFilePath)")
            return
        }

        // モデルのコンパイル
        let compiledModelURL = try MLModel.compileModel(at: URL(fileURLWithPath: modelFilePath))

        // コンパイルされたモデルファイルの存在確認
        guard fileManager.fileExists(atPath: compiledModelURL.path) else {
            XCTFail("コンパイルされたモデルファイルが存在しません: \(compiledModelURL.path)")
            throw ClassifierTestsError.modelFileMissing
        }

        // モデルの読み込み
        let model = try MLModel(contentsOf: compiledModelURL)
        let vnCoreMLModel = try VNCoreMLModel(for: model)
        let predictionRequest = VNCoreMLRequest(model: vnCoreMLModel)

        // クラスラベルの取得
        let classLabels = try fileManager.getClassLabelDirectories(resourcesPath: classifier.resourcesDirectoryPath)
            .map(\.lastPathComponent)
            .sorted()

        // 各クラスの画像で予測をテスト
        for classLabel in classLabels {
            let classDirURL = URL(fileURLWithPath: classifier.resourcesDirectoryPath)
                .appendingPathComponent(classLabel)
            let files = try fileManager.getFilesInDirectory(classDirURL)

            for file in files {
                let handler = VNImageRequestHandler(url: file, options: [:])
                try handler.perform([predictionRequest])

                guard let results = predictionRequest.results as? [VNClassificationObservation] else {
                    XCTFail("予測結果の型が不正です")
                    continue
                }

                XCTAssertFalse(results.isEmpty, "予測結果が空です")
                XCTAssertEqual(results.count, 2, "予測結果が2つではありません")

                let topResult = results[0]
                XCTAssertTrue(
                    classLabels.contains(topResult.identifier),
                    "予測結果のクラスラベル「\(topResult.identifier)」が期待されるクラスラベルに含まれていません"
                )
            }
        }
    }

    // モデルの再訓練をテスト
    func testModelRetraining() throws {
        // 1回目の訓練
        try classifier.createAndSaveModel(
            author: authorName,
            modelName: testModelName,
            version: testModelVersion,
            modelParameters: modelParameters
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
            modelParameters: modelParameters
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
        XCTAssertTrue(
            secondLatestResultDir.lastPathComponent > firstLatestResultDir.lastPathComponent,
            "2回目の結果ディレクトリが1回目より新しくありません"
        )
    }
}

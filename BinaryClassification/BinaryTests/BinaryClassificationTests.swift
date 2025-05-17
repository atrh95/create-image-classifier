@testable import BinaryClassification
import CoreML
import Foundation
import Vision
import XCTest

class BinaryClassificationTests: XCTestCase {
    var trainer: BinaryClassificationTrainer!
    let fileManager = FileManager.default
    var testModelName: String!
    var testModelVersion: String!
    var testResourcesRootPath: String!
    var temporaryOutputDirectoryURL: URL!
    var compiledModelURL: URL?
    var trainingResult: BinaryClassification.BinaryTrainingResult?
    var authorName: String!

    override func setUp() async throws {
        try await super.setUp()

        var sourcesDir = URL(fileURLWithPath: #filePath)
        sourcesDir.deleteLastPathComponent()
        testResourcesRootPath = sourcesDir.appendingPathComponent("TestResources").path

        temporaryOutputDirectoryURL = fileManager.temporaryDirectory
            .appendingPathComponent("TestOutput_\(UUID().uuidString)")
        print("一時ディレクトリを作成します: \(temporaryOutputDirectoryURL.path)")
        try fileManager.createDirectory(
            at: temporaryOutputDirectoryURL,
            withIntermediateDirectories: true,
            attributes: nil
        )

        trainer = BinaryClassificationTrainer(
            resourcesDirectoryPathOverride: testResourcesRootPath,
            outputDirectoryPathOverride: temporaryOutputDirectoryURL.path
        )
        testModelName = "TestCats_Binary_Run"
        testModelVersion = "\(Int(Date().timeIntervalSince1970))"
        authorName = "Test Author Setup"

        trainingResult = await trainer.train(
            author: authorName,
            modelName: testModelName,
            version: testModelVersion,
            maxIterations: 1
        )

        guard let result = trainingResult else {
            throw TestError.trainingFailed
        }

        let trainedModelURL = URL(fileURLWithPath: result.trainedModelFilePath)
        do {
            compiledModelURL = await try MLModel.compileModel(at: trainedModelURL)
        } catch {
            print("モデルのコンパイル失敗 in setUp: \(error.localizedDescription)")
            throw error
        }
    }

    override func tearDownWithError() throws {
        if let tempDir = temporaryOutputDirectoryURL, fileManager.fileExists(atPath: tempDir.path) {
            print("一時ディレクトリを削除します: \(tempDir.path)")
            try? fileManager.removeItem(at: tempDir)
        }
        temporaryOutputDirectoryURL = nil

        if let compiledUrl = compiledModelURL, fileManager.fileExists(atPath: compiledUrl.path) {
            try? fileManager.removeItem(at: compiledUrl)
        }
        compiledModelURL = nil
        trainingResult = nil
        trainer = nil
        try super.tearDownWithError()
    }

    func testTrainerInitialization() {
        XCTAssertNotNil(trainer, "BinaryClassificationTrainerの初期化失敗")
        XCTAssertEqual(trainer.resourcesDirectoryPath, testResourcesRootPath, "トレーナーのリソースパスがオーバーライド値と不一致")
        XCTAssertEqual(trainer.outputDirPath, temporaryOutputDirectoryURL.path, "トレーナーの出力パスがオーバーライド値と不一致")
    }

    enum TestError: Error {
        case testResourceMissing
        case trainingFailed
        case modelFileMissing
        case predictionFailed
        case setupFailed
    }

    func testModelTrainingAndArtifacts() throws {
        guard let result = trainingResult else {
            XCTFail("訓練結果がnil (testModelTrainingAndArtifacts)")
            throw TestError.trainingFailed
        }

        XCTAssertTrue(
            fileManager.fileExists(atPath: result.trainedModelFilePath),
            "訓練モデルファイルが期待パス「\(result.trainedModelFilePath)」に見つからない"
        )

        let expectedClassLabels = ["NotScary", "Scary"].sorted()
        XCTAssertEqual(
            Set(result.detectedClassLabelsList.sorted()),
            Set(expectedClassLabels),
            "検出クラスラベル「\(result.detectedClassLabelsList.sorted())」が期待ラベル「\(expectedClassLabels)」と不一致"
        )

        result.saveLog(modelAuthor: authorName, modelName: testModelName, modelVersion: testModelVersion)
        let modelFileDir = URL(fileURLWithPath: result.trainedModelFilePath).deletingLastPathComponent()
        let expectedLogFileName = "\(testModelName!)_\(testModelVersion!).md"
        let expectedLogFilePath = modelFileDir.appendingPathComponent(expectedLogFileName).path
        XCTAssertTrue(fileManager.fileExists(atPath: expectedLogFilePath), "ログファイルが期待パス「\(expectedLogFilePath)」に未生成")

        XCTAssertEqual(result.modelName, testModelName, "訓練結果modelName「\(result.modelName)」が期待値「\(testModelName!)」と不一致")
        XCTAssertEqual(result.maxIterations, 1, "訓練結果maxIterations「\(result.maxIterations)」が期待値「1」と不一致")

        do {
            let logContents = try String(contentsOfFile: expectedLogFilePath, encoding: .utf8)
            XCTAssertFalse(logContents.isEmpty, "ログファイルが空です: \(expectedLogFilePath)")
        } catch {
            XCTFail("ログファイル内容読込不可: \(expectedLogFilePath), エラー: \(error.localizedDescription)")
        }

        XCTAssertTrue(
            result.trainedModelFilePath.contains(testModelName),
            "モデルファイルパスにモデル名「\(testModelName!)」不足。パス: \(result.trainedModelFilePath)"
        )
        XCTAssertTrue(
            result.trainedModelFilePath.contains(testModelVersion),
            "モデルファイルパスにバージョン「\(testModelVersion!)」不足。パス: \(result.trainedModelFilePath)"
        )
        XCTAssertTrue(
            result.trainedModelFilePath.contains(trainer.classificationMethod),
            "モデルファイルパスに分類メソッド「\(trainer.classificationMethod)」不足。パス: \(result.trainedModelFilePath)"
        )
    }

    func testModelPredictionAccuracy() throws {
        guard let finalModelURL = compiledModelURL else {
            XCTFail("コンパイル済みモデルURLがnil (testModelPredictionAccuracy)")
            throw TestError.modelFileMissing
        }

        let mlModel = try MLModel(contentsOf: finalModelURL)
        let vnCoreMLModel = try VNCoreMLModel(for: mlModel)
        let predictionRequest = VNCoreMLRequest(model: vnCoreMLModel)

        let bundle = Bundle(for: type(of: self))
        print("Bundle Path: \(bundle.bundlePath)")
        let expectedScaryURL = bundle.url(forResource: "cat_vh", withExtension: "jpg", subdirectory: "Scary")
        print("Attempted Scary Image URL: \(expectedScaryURL?.path ?? "nil")")
        let expectedNotScaryURL = bundle.url(forResource: "cat_pb", withExtension: "jpg", subdirectory: "NotScary")
        print("Attempted NotScary Image URL: \(expectedNotScaryURL?.path ?? "nil")")

        guard let scaryTestImageURL = expectedScaryURL,
            let notScaryTestImageURL = expectedNotScaryURL
        else {
            XCTFail("予測用テスト画像URL発見不可")
            throw TestError.testResourceMissing
        }

        let scaryImageHandler = VNImageRequestHandler(url: scaryTestImageURL, options: [:])
        try scaryImageHandler.perform([predictionRequest])
        guard let scaryObservations = predictionRequest.results as? [VNClassificationObservation],
              let scaryTopResult = scaryObservations.first
        else {
            XCTFail("Scary画像: 有効な分類結果オブジェクトを取得できませんでした。")
            throw TestError.predictionFailed
        }
        XCTAssertNotNil(scaryTopResult.identifier, "Scary画像: 予測結果からクラスラベルを取得できませんでした。")

        let notScaryImageHandler = VNImageRequestHandler(url: notScaryTestImageURL, options: [:])
        try notScaryImageHandler.perform([predictionRequest])
        guard let notScaryObservations = predictionRequest.results as? [VNClassificationObservation],
              let notScaryTopResult = notScaryObservations.first
        else {
            XCTFail("NotScary画像: 有効な分類結果オブジェクトを取得できませんでした。")
            throw TestError.predictionFailed
        }
        XCTAssertNotNil(notScaryTopResult.identifier, "NotScary画像: 予測結果からクラスラベルを取得できませんでした。")
    }
}

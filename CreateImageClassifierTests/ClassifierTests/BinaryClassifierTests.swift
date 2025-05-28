@testable import BinaryClassification
import CICFileManager
import CoreML
import CreateML
import Foundation
import Vision
import XCTest

final class BinaryClassifierTests: XCTestCase {
    var classifier: BinaryClassifier!
    let fileManager = FileManager.default
    var authorName: String = "Test Author"
    var testModelName: String = "TestModel_Binary_Run"
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
    var compiledModelURL: URL?
    var trainingResult: BinaryClassification.BinaryTrainingResult?

    override func setUp() async throws {
        try await super.setUp()

        temporaryOutputDirectoryURL = fileManager.temporaryDirectory
            .appendingPathComponent("TestOutput_Binary")
        try fileManager.createDirectory(
            at: temporaryOutputDirectoryURL,
            withIntermediateDirectories: true,
            attributes: nil
        )

        classifier = BinaryClassifier(
            outputDirectoryPathOverride: temporaryOutputDirectoryURL.path
        )
        
        // テストリソースディレクトリのパスを設定
        let currentFileURL = URL(fileURLWithPath: #filePath)
        classifier.testResourcesDirectoryPath = currentFileURL
            .deletingLastPathComponent() // BinaryClassificationTests.swift
            .appendingPathComponent("TestResources")
            .appendingPathComponent("Binary")
            .path

        trainingResult = await classifier.train(
            author: "test",
            modelName: testModelName,
            version: "v1",
            modelParameters: modelParameters,
            scenePrintRevision: nil
        )

        guard let result = trainingResult else {
            throw TestError.trainingFailed
        }

        let trainedModelURL = URL(fileURLWithPath: result.metadata.trainedModelFilePath)
        do {
            compiledModelURL = try await MLModel.compileModel(at: trainedModelURL)
        } catch {
            throw error
        }
    }

    override func tearDownWithError() throws {
        if let tempDir = temporaryOutputDirectoryURL, fileManager.fileExists(atPath: tempDir.path) {
            try? fileManager.removeItem(at: tempDir)
        }
        temporaryOutputDirectoryURL = nil

        if let compiledUrl = compiledModelURL, fileManager.fileExists(atPath: compiledUrl.path) {
            try? fileManager.removeItem(at: compiledUrl)
        }
        compiledModelURL = nil
        trainingResult = nil
        classifier.testResourcesDirectoryPath = nil
        classifier = nil
        try super.tearDownWithError()
    }

    func testClassifierDIConfiguration() {
        XCTAssertNotNil(classifier, "BinaryClassifierの初期化失敗")
        XCTAssertEqual(classifier.outputDirPath, temporaryOutputDirectoryURL.path, "分類器の出力パスが期待値と不一致")
    }

    enum TestError: Error {
        case testResourceMissing
        case trainingFailed
        case modelFileMissing
        case predictionFailed
        case setupFailed
    }

    // モデルの訓練と成果物の生成をテスト
    func testModelTrainingAndArtifactGeneration() throws {
        guard let result = trainingResult else {
            XCTFail("訓練結果がnilです (testModelTrainingAndArtifactGeneration)")
            throw TestError.trainingFailed
        }

        XCTAssertTrue(
            fileManager.fileExists(atPath: result.modelOutputPath),
            "訓練モデルファイルが期待されるパス「\(result.modelOutputPath)」に見つかりません"
        )

        // Dynamically get expected class labels from the TestResources subdirectories
        let resourceURL = URL(fileURLWithPath: classifier.resourcesDirectoryPath)

        var expectedClassLabels: [String] = []
        do {
            let subdirectories = try fileManager.contentsOfDirectory(
                at: resourceURL,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles, .skipsSubdirectoryDescendants]
            )
            expectedClassLabels = subdirectories.filter { url in
                var isDirectory: ObjCBool = false
                return fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory) && isDirectory.boolValue
            }.map(\.lastPathComponent).sorted()
        } catch {
            XCTFail("テストリソースのサブディレクトリからのクラスラベルの取得に失敗しました: \(error.localizedDescription)")
            throw TestError.setupFailed
        }

        guard !expectedClassLabels.isEmpty else {
            XCTFail("テストリソースディレクトリから期待されるクラスラベルが見つかりませんでした。パス: \(resourceURL.path)")
            throw TestError.setupFailed
        }

        XCTAssertEqual(
            Set(result.metadata.detectedClassLabelsList.sorted()),
            Set(expectedClassLabels),
            "検出されたクラスラベル「\(result.metadata.detectedClassLabelsList.sorted())」が期待されるラベル「\(expectedClassLabels)」と一致しません"
        )

        result.saveLog(modelAuthor: authorName, modelName: testModelName, modelVersion: testModelVersion)
        let modelFileDir = URL(fileURLWithPath: result.metadata.trainedModelFilePath).deletingLastPathComponent()

        let expectedLogFileName = "Binary_Run_Report_\(testModelVersion).md"
        let expectedLogFilePath = modelFileDir.appendingPathComponent(expectedLogFileName).path
        XCTAssertTrue(fileManager.fileExists(atPath: expectedLogFilePath), "ログファイルが期待パス「\(expectedLogFilePath)」に未生成")

        XCTAssertEqual(
            result.metadata.modelName,
            testModelName,
            "訓練結果modelName「\(result.metadata.modelName)」が期待値「\(testModelName)」と不一致"
        )

        do {
            let logContents = try String(contentsOfFile: expectedLogFilePath, encoding: .utf8)
            XCTAssertFalse(logContents.isEmpty, "ログファイルが空です: \(expectedLogFilePath)")
        } catch {
            XCTFail("ログファイル内容読込不可: \(expectedLogFilePath), エラー: \(error.localizedDescription)")
        }

        XCTAssertTrue(
            result.metadata.trainedModelFilePath.contains(testModelName),
            "モデルファイルパスにモデル名「\(testModelName)」が含まれていません"
        )
        XCTAssertTrue(
            result.metadata.trainedModelFilePath.contains(testModelVersion),
            "モデルファイルパスにバージョン「\(testModelVersion)」が含まれていません"
        )
        XCTAssertTrue(
            result.metadata.trainedModelFilePath.contains(classifier.classificationMethod),
            "モデルファイルパスに分類法「\(classifier.classificationMethod)」が含まれていません"
        )
    }

    // モデルが予測を実行できるかテスト
    func testModelCanPerformPrediction() throws {
        guard let finalModelURL = compiledModelURL else {
            XCTFail("コンパイル済みモデルURLがnil (testModelPredictionAccuracy)")
            throw TestError.modelFileMissing
        }
        guard let result = trainingResult else {
            XCTFail("訓練結果がnil (testModelPredictionAccuracy)")
            throw TestError.trainingFailed
        }

        let mlModel = try MLModel(contentsOf: finalModelURL)
        let vnCoreMLModel = try VNCoreMLModel(for: mlModel)
        let predictionRequest = VNCoreMLRequest(model: vnCoreMLModel)

        let classLabels = result.metadata.detectedClassLabelsList.sorted()
        guard classLabels.count >= 2 else {
            XCTFail("テストには少なくとも2つのクラスラベルが訓練結果に必要です。検出されたラベル: \(classLabels)")
            throw TestError.setupFailed
        }

        let classLabel1 = classLabels[0]
        let classLabel2 = classLabels[1]

        print("動的に識別されたテスト用クラスラベル: '\(classLabel1)' および '\(classLabel2)'")

        // classLabel1 の画像取得処理
        let imageURL1: URL
        do {
            imageURL1 = try getRandomImageURL(forClassLabel: classLabel1)
        } catch {
            XCTFail("'\(classLabel1)' サブディレクトリからのランダム画像取得失敗。エラー: \(error.localizedDescription)")
            throw error
        }

        // classLabel2 の画像取得処理
        let imageURL2: URL
        do {
            imageURL2 = try getRandomImageURL(forClassLabel: classLabel2)
        } catch {
            XCTFail("'\(classLabel2)' サブディレクトリからのランダム画像取得失敗。エラー: \(error.localizedDescription)")
            throw error
        }

        let imageHandler1 = VNImageRequestHandler(url: imageURL1, options: [:])
        try imageHandler1.perform([predictionRequest])
        guard let observations1 = predictionRequest.results as? [VNClassificationObservation],
              let topResult1 = observations1.first
        else {
            XCTFail("クラス '\(classLabel1)' 画像: 有効な分類結果オブジェクトを取得できませんでした。")
            throw TestError.predictionFailed
        }
        XCTAssertNotNil(topResult1.identifier, "クラス '\(classLabel1)' 画像: 予測結果からクラスラベルを取得できませんでした。")
        print("クラス '\(classLabel1)' 画像の予測 (正解ラベル): \(topResult1.identifier) (確信度: \(topResult1.confidence))")

        let imageHandler2 = VNImageRequestHandler(url: imageURL2, options: [:])
        try imageHandler2.perform([predictionRequest])
        guard let observations2 = predictionRequest.results as? [VNClassificationObservation],
              let topResult2 = observations2.first
        else {
            XCTFail("クラス '\(classLabel2)' 画像: 有効な分類結果オブジェクトを取得できませんでした。")
            throw TestError.predictionFailed
        }
        XCTAssertNotNil(topResult2.identifier, "クラス '\(classLabel2)' 画像: 予測結果からクラスラベルを取得できませんでした。")
        print("クラス '\(classLabel2)' 画像の予測 (正解ラベル): \(topResult2.identifier) (確信度: \(topResult2.confidence))")
    }

    private func getRandomImageURL(forClassLabel classLabel: String) throws -> URL {
        let resourceURL = URL(fileURLWithPath: classifier.resourcesDirectoryPath)
        let classLabelURL = resourceURL.appendingPathComponent(classLabel)

        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: classLabelURL.path, isDirectory: &isDirectory),
              isDirectory.boolValue
        else {
            let message = "サブディレクトリ '\(classLabel)' が見つからないか、ディレクトリではありません: \(classLabelURL.path)"
            XCTFail(message)
            throw TestError.testResourceMissing
        }

        let validExtensions = ["jpg", "jpeg", "png"]
        let allFiles = try fileManager.contentsOfDirectory(
            at: classLabelURL,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        ).filter { url in
            validExtensions.contains(url.pathExtension.lowercased())
        }

        guard !allFiles.isEmpty else {
            throw TestError.testResourceMissing
        }

        return allFiles.randomElement()!
    }
}

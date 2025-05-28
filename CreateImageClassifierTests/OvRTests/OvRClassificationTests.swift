import CoreML
import CreateML
import Foundation
@testable import OvRClassification
import Vision
import XCTest

final class OvRClassificationTests: XCTestCase {
    var trainer: OvRClassificationTrainer!
    let fileManager = FileManager.default
    let authorName: String = "Test Author"
    let testModelName: String = "TestCats_OvR_Run"
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

    var testResourcesRootPath: String {
        var currentTestFileDir = URL(fileURLWithPath: #filePath)
        currentTestFileDir.deleteLastPathComponent()
        return currentTestFileDir.appendingPathComponent("TestResources").path
    }

    var temporaryOutputDirectoryURL: URL!
    var compiledModelURL: URL?
    var trainingResult: OvRClassification.OvRTrainingResult?

    enum TestError: Error {
        case trainingFailed
        case modelFileMissing
        case predictionFailed
        case setupFailed
        case resourcePathError
    }

    override func setUp() async throws {
        try await super.setUp()

        temporaryOutputDirectoryURL = fileManager.temporaryDirectory
            .appendingPathComponent("TestOutput_OvR")
        try fileManager.createDirectory(
            at: temporaryOutputDirectoryURL,
            withIntermediateDirectories: true,
            attributes: nil
        )

        trainer = OvRClassificationTrainer(
            resourcesDirectoryPathOverride: testResourcesRootPath,
            outputDirectoryPathOverride: temporaryOutputDirectoryURL.path
        )

        trainingResult = await trainer.train(
            author: authorName,
            modelName: testModelName,
            version: testModelVersion,
            modelParameters: modelParameters,
            scenePrintRevision: 1
        )

        guard trainingResult != nil else {
            throw TestError.trainingFailed
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
        trainer = nil
        try super.tearDownWithError()
    }

    func testTrainerDIConfiguration() throws {
        XCTAssertNotNil(trainer, "OvRClassificationTrainerの初期化失敗")
        XCTAssertEqual(trainer.resourcesDirectoryPath, testResourcesRootPath, "トレーナーのリソースパスが期待値と不一致")
        XCTAssertEqual(trainer.outputDirPath, temporaryOutputDirectoryURL.path, "トレーナーの出力パスが期待値と不一致")
    }

    func testModelTrainingAndArtifactGeneration() throws {
        guard let result = trainingResult else {
            XCTFail("訓練結果がnilです (testModelTrainingAndArtifactGeneration)")
            throw TestError.trainingFailed
        }

        XCTAssertTrue(
            fileManager.fileExists(atPath: result.metadata.trainedModelFilePath),
            "訓練モデルファイルが期待されるパス「\(result.metadata.trainedModelFilePath)」に見つかりません"
        )

        let expectedClassLabels = ["black_and_white", "human_hands_detected", "mouth_open", "safe", "sphynx"].sorted()
        XCTAssertEqual(
            Set(result.metadata.detectedClassLabelsList.sorted()),
            Set(expectedClassLabels),
            "訓練結果のクラスラベル「\(result.metadata.detectedClassLabelsList.sorted())」が期待されるラベル「\(expectedClassLabels)」と一致しません"
        )

        result.saveLog(modelAuthor: authorName, modelName: testModelName, modelVersion: testModelVersion)
        let modelFileDir = URL(fileURLWithPath: result.metadata.trainedModelFilePath).deletingLastPathComponent()
        let expectedLogFileName = "OvR_Run_Report_\(testModelVersion).md"
        let expectedLogFilePath = modelFileDir.appendingPathComponent(expectedLogFileName).path
        XCTAssertTrue(fileManager.fileExists(atPath: expectedLogFilePath), "ログファイル「\(expectedLogFilePath)」が生成されていません")

        XCTAssertEqual(result.metadata.modelName, testModelName)
        XCTAssertFalse(result.metadata.sourceTrainingDataDirectoryPath.isEmpty, "訓練データパスが空です")
    }

    func testModelCanPerformPrediction() async throws {
        guard let result = trainingResult else {
            XCTFail("訓練結果がnil (OvR testModelCanPerformPrediction)")
            throw TestError.trainingFailed
        }

        let modelFilePath = result.metadata.trainedModelFilePath
        guard fileManager.fileExists(atPath: modelFilePath) else {
            XCTFail("モデルファイルが見つかりません: \(modelFilePath)")
            throw TestError.modelFileMissing
        }

        print("Attempting to compile model: \(modelFilePath)")
        let specificCompiledModelURL: URL
        do {
            specificCompiledModelURL = try await MLModel.compileModel(at: URL(fileURLWithPath: modelFilePath))
            compiledModelURL = specificCompiledModelURL
        } catch {
            XCTFail("モデルのコンパイル失敗 (\(modelFilePath)): \(error.localizedDescription)")
            throw TestError.modelFileMissing
        }

        let imageURL: URL
        do {
            imageURL =
                try getRandomImageURLFromTestResources(inBaseDirectory: URL(fileURLWithPath: testResourcesRootPath))
        } catch {
            XCTFail("テストリソースからのランダム画像取得失敗。エラー: \(error.localizedDescription)")
            throw error
        }

        print("Test image for prediction: \(imageURL.path)")

        guard fileManager.fileExists(atPath: imageURL.path) else {
            XCTFail("OvRテスト用画像ファイルが見つかりません: \(imageURL.path)")
            throw TestError.resourcePathError
        }

        let mlModel = try MLModel(contentsOf: specificCompiledModelURL)
        let visionModel = try VNCoreMLModel(for: mlModel)
        let request = VNCoreMLRequest(model: visionModel) { request, error in
            if let error {
                XCTFail("VNCoreMLRequest failed: \(error.localizedDescription)")
                return
            }
            guard let observations = request.results as? [VNClassificationObservation] else {
                XCTFail("予測結果をVNClassificationObservationにキャストできませんでした。")
                return
            }

            XCTAssertFalse(observations.isEmpty, "予測結果(observations)が空でした。モデルは予測を行いませんでした。")

            if let topResult = observations.first {
                print(
                    "OvR Top prediction for \(imageURL.lastPathComponent) using \(URL(fileURLWithPath: modelFilePath).lastPathComponent): \(topResult.identifier) with confidence \(topResult.confidence)"
                )
            } else {
                print(
                    "OvR prediction for \(imageURL.lastPathComponent) using \(URL(fileURLWithPath: modelFilePath).lastPathComponent): No observations found, though this should have been caught by XCTAssertFalse."
                )
            }
        }

        let handler = VNImageRequestHandler(url: imageURL, options: [:])
        try handler.perform([request])
    }

    private func getRandomImageURLFromTestResources(inBaseDirectory baseDirectoryURL: URL) throws -> URL {
        let allLabelDirectories = try fileManager.contentsOfDirectory(
            at: baseDirectoryURL,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: .skipsHiddenFiles
        ).filter { url in
            var isDirectory: ObjCBool = false
            fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory)
            return isDirectory.boolValue && !url.lastPathComponent.hasPrefix(".")
        }

        guard !allLabelDirectories.isEmpty else {
            let message = "テストリソースディレクトリ「\(baseDirectoryURL.path)」内にラベルサブディレクトリが見つかりません。"
            XCTFail(message)
            throw TestError.resourcePathError
        }

        guard let randomLabelDirURL = allLabelDirectories.randomElement() else {
            let message = "ランダムなラベルディレクトリの選択に失敗しました。"
            XCTFail(message)
            throw TestError.resourcePathError
        }

        print("Randomly selected label directory for OvR test image: \(randomLabelDirURL.path)")

        let imageFiles = try fileManager.contentsOfDirectory(
            at: randomLabelDirURL,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles, .skipsSubdirectoryDescendants]
        ).filter { url in
            let pathExtension = url.pathExtension.lowercased()
            return ["jpg", "jpeg", "png", "heic"].contains(pathExtension) && !url.lastPathComponent.hasPrefix(".")
        }

        guard let randomImageURL = imageFiles.randomElement() else {
            let message = "ラベルディレクトリ「\(randomLabelDirURL.path)」内に画像ファイルが見つかりません。"
            XCTFail(message)
            throw TestError.resourcePathError
        }

        return randomImageURL
    }
}

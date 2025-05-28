import CoreML
import CreateML
import Foundation
@testable import MultiLabelClassification
import Vision
import XCTest

final class MultiLabelClassificationTests: XCTestCase {
    var trainer: MultiLabelClassificationTrainer!
    let fileManager = FileManager.default
    let authorName: String = "Test Author"
    let testModelName: String = "TestCats_MultiLabel_Run"
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

    var testAnnotationFilePath: String {
        URL(fileURLWithPath: testResourcesRootPath).appendingPathComponent("multilabel_cat_annotations.json").path
    }

    var temporaryOutputDirectoryURL: URL!
    var compiledModelURL: URL?
    var trainingResult: MultiLabelClassification.MultiLabelTrainingResult?

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
            .appendingPathComponent("TestOutput_MultiLabel")
        try fileManager.createDirectory(
            at: temporaryOutputDirectoryURL,
            withIntermediateDirectories: true,
            attributes: nil
        )

        trainer = MultiLabelClassificationTrainer(
            resourcesDirectoryPathOverride: testResourcesRootPath,
            outputDirectoryPathOverride: temporaryOutputDirectoryURL.path,
            annotationFilePathOverride: testAnnotationFilePath
        )

        trainingResult = await trainer.train(
            author: authorName,
            modelName: testModelName,
            version: testModelVersion,
            modelParameters: modelParameters,
            scenePrintRevision: 1
        )

        guard let result = trainingResult else {
            throw TestError.trainingFailed
        }

        let trainedModelURL = URL(fileURLWithPath: result.modelOutputPath)
        do {
            compiledModelURL = try await MLModel.compileModel(at: trainedModelURL)
        } catch {
            print("モデルのコンパイル失敗 (setUp): \(error.localizedDescription)")
            throw TestError.modelFileMissing
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

    func testTrainerDIConfiguration() {
        XCTAssertNotNil(trainer, "MultiLabelClassificationTrainerの初期化失敗")
        XCTAssertEqual(trainer.resourcesDirectoryPath, testResourcesRootPath, "トレーナーのリソースパスが期待値と不一致")
        XCTAssertEqual(trainer.outputDirPath, temporaryOutputDirectoryURL.path, "トレーナーの出力パスが期待値と不一致")
    }

    func testModelTrainingAndArtifactGeneration() throws {
        guard let result = trainingResult else {
            XCTFail("訓練結果がnilです (testModelTrainingAndArtifactGeneration)")
            throw TestError.trainingFailed
        }

        XCTAssertTrue(
            fileManager.fileExists(atPath: result.modelOutputPath),
            "訓練モデルファイルが期待されるパス「\(result.modelOutputPath)」に見つかりません"
        )

        let expectedClassLabels = ["black_and_white", "human_hands_detected", "mouth_open", "safe", "sphynx"].sorted()
        XCTAssertEqual(
            Set(result.classLabels.sorted()),
            Set(expectedClassLabels),
            "訓練結果のクラスラベル「\(result.classLabels.sorted())」が期待されるラベル「\(expectedClassLabels)」と一致しません"
        )

        result.saveLog(modelAuthor: authorName, modelName: testModelName, modelVersion: testModelVersion)
        let modelFileDir = URL(fileURLWithPath: result.modelOutputPath).deletingLastPathComponent()
        let expectedLogFileName = "\(testModelName)_\(testModelVersion).md"
        let expectedLogFilePath = modelFileDir.appendingPathComponent(expectedLogFileName).path
        XCTAssertTrue(fileManager.fileExists(atPath: expectedLogFilePath), "ログファイル「\(expectedLogFilePath)」が生成されていません")

        XCTAssertEqual(result.modelName, testModelName)
        XCTAssertFalse(result.trainingDataPath.isEmpty, "訓練データパスが空です")
    }

    func testModelCanPerformPrediction() throws {
        guard let finalModelURL = compiledModelURL else {
            XCTFail("コンパイル済みモデルのURLがnil (testModelCanPerformPrediction)")
            throw TestError.modelFileMissing
        }
        guard trainingResult != nil else {
            XCTFail("訓練結果がnil (testModelCanPerformPrediction)")
            throw TestError.trainingFailed
        }

        // テスト用の画像ファイルを取得
        let resourcesURL = URL(fileURLWithPath: testResourcesRootPath)
        let classDirs = try fileManager.contentsOfDirectory(at: resourcesURL, includingPropertiesForKeys: nil)
        guard let firstClassDir = classDirs.first,
              let imageFile = try fileManager.contentsOfDirectory(at: firstClassDir, includingPropertiesForKeys: nil).first
        else {
            XCTFail("テスト用画像ファイルが見つかりません")
            throw TestError.resourcePathError
        }

        let mlModel = try MLModel(contentsOf: finalModelURL)
        let visionModel = try VNCoreMLModel(for: mlModel)
        let request = VNCoreMLRequest(model: visionModel)

        let handler = VNImageRequestHandler(url: imageFile, options: [:])
        try handler.perform([request])

        guard let observations = request.results as? [VNClassificationObservation] else {
            XCTFail("予測結果をVNClassificationObservationにキャストできませんでした。")
            throw TestError.predictionFailed
        }

        XCTAssertFalse(observations.isEmpty, "予測結果(observations)が空でした。モデルは予測を行いませんでした。")

        if !observations.isEmpty {
            let allPredictedLabelsWithConfidence = observations
                .map { "\($0.identifier) (信頼度: \(String(format: "%.2f", $0.confidence)))" }.joined(separator: ", ")
            print("ファイル「\(imageFile.lastPathComponent)」の全予測ラベル: [\(allPredictedLabelsWithConfidence)]")
        }
    }
}

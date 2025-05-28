import CoreML
import CreateML
import Foundation
@testable import OvRClassification
import Vision
import XCTest
import CICFileManager

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

    var temporaryOutputDirectoryURL: URL!
    var compiledModelURL: URL?
    var trainingResult: OvRClassification.OvRTrainingResult?

    enum TestError: Error {
        case trainingFailed
        case modelFileMissing
        case predictionFailed
        case setupFailed
        case resourcePathError
        case testResourceMissing
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

        // Dynamically get expected class labels from the TestResources subdirectories
        let resourceURL = URL(fileURLWithPath: trainer.resourcesDirectoryPath)

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
            imageURL = try getRandomImageURL(forClassLabel: "black_and_white")
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

    private func getRandomImageURL(forClassLabel classLabel: String) throws -> URL {
        let resourceURL = URL(fileURLWithPath: trainer.resourcesDirectoryPath)
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

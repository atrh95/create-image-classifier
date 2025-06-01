import CICFileManager
import CoreML
import CreateML
import Foundation
@testable import MultiLabelClassification
import Vision
import XCTest

final class MultiLabelClassifierTests: XCTestCase {
    var classifier: MultiLabelClassifier!
    let fileManager = FileManager.default
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
            .appendingPathComponent("TestOutput_MultiLabel")
        try fileManager.createDirectory(
            at: temporaryOutputDirectoryURL,
            withIntermediateDirectories: true,
            attributes: nil
        )

        let resourceDirectoryPath = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // MultiLabelClassifierTests
            .deletingLastPathComponent() // Tests
            .appendingPathComponent("TestResources")
            .appendingPathComponent("MultiLabel")
            .path

        classifier = MultiLabelClassifier(
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

    func testClassifierDIConfiguration() async throws {
        // モデルの作成
        let result = try await classifier.create(
            author: authorName,
            modelName: testModelName,
            version: testModelVersion,
            modelParameters: modelParameters
        )
        
        XCTAssertNotNil(classifier, "MultiLabelClassifierの初期化失敗")
        XCTAssertEqual(classifier.outputParentDirPath, temporaryOutputDirectoryURL.path, "分類器の出力パスが期待値と不一致")
    }

    func testModelTrainingAndArtifactGeneration() async throws {
        // モデルの作成
        let result = try await classifier.create(
            author: authorName,
            modelName: testModelName,
            version: testModelVersion,
            modelParameters: modelParameters
        )

        XCTAssertTrue(
            fileManager.fileExists(atPath: result.modelOutputPath),
            "訓練モデルファイルが期待されるパス「\(result.modelOutputPath)」に見つかりません"
        )

        let expectedClassLabels = ["black_and_white", "human_hands_detected", "mouth_open", "sphynx"].sorted()
        XCTAssertEqual(
            Set(result.metadata.detectedClassLabelsList.sorted()),
            Set(expectedClassLabels),
            "訓練結果のクラスラベル「\(result.metadata.detectedClassLabelsList.sorted())」が期待されるラベル「\(expectedClassLabels)」と一致しません"
        )

        result.saveLog(modelAuthor: authorName, modelName: testModelName, modelVersion: testModelVersion)
        let modelFileDir = URL(fileURLWithPath: result.modelOutputPath).deletingLastPathComponent()
        let resultDir = modelFileDir.appendingPathComponent("MultiLabel_Result_1")
        let expectedLogFileName = "MultiLabel_Run_Report_\(testModelVersion).md"
        let expectedLogFilePath = resultDir.appendingPathComponent(expectedLogFileName).path
        XCTAssertTrue(fileManager.fileExists(atPath: expectedLogFilePath), "ログファイル「\(expectedLogFilePath)」が生成されていません")

        XCTAssertEqual(result.metadata.modelName, testModelName)

        // モデルファイル名の検証
        let modelFilePath = result.metadata.trainedModelFilePath
        let modelFileName = URL(fileURLWithPath: modelFilePath).lastPathComponent
        let regex = #"^TestModel_MultiLabel_v\d+\.mlmodel$"#
        XCTAssertTrue(
            modelFileName.range(of: regex, options: .regularExpression) != nil,
            """
            モデルファイル名が期待パターンに一致しません。
            期待パターン: \(regex)
            実際: \(modelFileName)
            """
        )

        XCTAssertTrue(
            result.metadata.trainedModelFilePath.contains(testModelVersion),
            "モデルファイルのパスにバージョン「\(testModelVersion)」が含まれていません"
        )
        XCTAssertTrue(
            result.metadata.trainedModelFilePath.contains(classifier.classificationMethod),
            "モデルファイルのパスに分類手法「\(classifier.classificationMethod)」が含まれていません"
        )
    }

    // モデルが予測を実行できるかテスト
    func testModelCanPerformPrediction() async throws {
        // モデルの作成
        let result = try await classifier.create(
            author: authorName,
            modelName: testModelName,
            version: testModelVersion,
            modelParameters: modelParameters
        )

        // モデルのコンパイル
        let trainedModelURL = URL(fileURLWithPath: result.metadata.trainedModelFilePath)
        let compiledModelURL = try await MLModel.compileModel(at: trainedModelURL)

        // コンパイルされたモデルファイルの存在確認
        guard fileManager.fileExists(atPath: compiledModelURL.path) else {
            XCTFail("コンパイルされたモデルファイルが存在しません: \(compiledModelURL.path)")
            throw ClassifierTestsError.modelFileMissing
        }

        // モデルの読み込みを試みる
        let mlModel: MLModel
        do {
            mlModel = try MLModel(contentsOf: compiledModelURL)
        } catch {
            XCTFail("モデルの読み込みに失敗しました: \(error.localizedDescription)")
            throw error
        }

        let visionModel = try VNCoreMLModel(for: mlModel)
        let request = VNCoreMLRequest(model: visionModel)

        let imageURL = try getRandomImageURL(forClassLabel: "black_and_white")
        let handler = VNImageRequestHandler(url: imageURL, options: [:])
        try handler.perform([request])

        guard let observations = request.results as? [VNClassificationObservation] else {
            XCTFail("予測結果をVNClassificationObservationにキャストできませんでした。")
            throw ClassifierTestsError.predictionFailed
        }

        XCTAssertFalse(observations.isEmpty, "予測結果(observations)が空でした。モデルは予測を行いませんでした。")

        if !observations.isEmpty {
            let allPredictedLabelsWithConfidence = observations
                .map { "\($0.identifier) (信頼度: \(String(format: "%.2f", $0.confidence)))" }.joined(separator: ", ")
            print("ファイル「\(imageURL.lastPathComponent)」の全予測ラベル: [\(allPredictedLabelsWithConfidence)]")
        }
    }

    // 出力ディレクトリの連番を検証
    func testSequentialOutputDirectoryNumbering() async throws {
        // 1回目のモデル作成
        let firstResult = try await classifier.create(
            author: authorName,
            modelName: testModelName,
            version: testModelVersion,
            modelParameters: modelParameters
        )

        let firstModelFileDir = URL(fileURLWithPath: firstResult.metadata.trainedModelFilePath)
            .deletingLastPathComponent()

        // 2回目のモデル作成を実行
        let secondResult = try await classifier.create(
            author: "TestAuthor",
            modelName: testModelName,
            version: "v1",
            modelParameters: modelParameters
        )

        let secondModelFileDir = URL(fileURLWithPath: secondResult.metadata.trainedModelFilePath)
            .deletingLastPathComponent()

        // 連番の検証
        let firstResultNumber = Int(firstModelFileDir.lastPathComponent.replacingOccurrences(
            of: "MultiLabel_Result_",
            with: ""
        )) ?? 0
        let secondResultNumber = Int(secondModelFileDir.lastPathComponent.replacingOccurrences(
            of: "MultiLabel_Result_",
            with: ""
        )) ?? 0
        XCTAssertEqual(
            secondResultNumber,
            firstResultNumber + 1,
            "2回目の出力ディレクトリの連番が期待値と一致しません。\n1回目: \(firstResultNumber)\n2回目: \(secondResultNumber)"
        )
    }

    private func getRandomImageURL(forClassLabel classLabel: String) throws -> URL {
        try TestUtils.getRandomImageURL(
            forClassLabel: classLabel,
            resourcesDirectoryPath: classifier.resourcesDirectoryPath,
            fileManager: fileManager
        )
    }
}

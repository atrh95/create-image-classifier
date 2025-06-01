import BinaryClassification
import CICFileManager
import CoreML
import CreateML
import Foundation
@testable import MultiClassClassification
import MultiLabelClassification
import OvOClassification
import OvRClassification
import Vision
import XCTest

final class MultiClassClassifierTests: XCTestCase {
    var classifier: MultiClassClassifier!
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
            .appendingPathComponent("TestOutput_MultiClass")
        try fileManager.createDirectory(
            at: temporaryOutputDirectoryURL,
            withIntermediateDirectories: true,
            attributes: nil
        )

        let resourceDirectoryPath = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // MultiClassClassifierTests
            .deletingLastPathComponent() // Tests
            .appendingPathComponent("TestResources")
            .appendingPathComponent("MultiClass")
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

    func testClassifierDIConfiguration() async throws {
        // モデルの作成
        let result = try await classifier.create(
            author: authorName,
            modelName: testModelName,
            version: testModelVersion,
            modelParameters: modelParameters
        )
        
        XCTAssertNotNil(classifier, "MultiClassClassifierの初期化失敗")
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
            throw ClassifierTestsError.setupFailed
        }

        guard !expectedClassLabels.isEmpty else {
            XCTFail("テストリソースディレクトリから期待されるクラスラベルが見つかりませんでした。パス: \(resourceURL.path)")
            throw ClassifierTestsError.setupFailed
        }

        XCTAssertEqual(
            Set(result.metadata.detectedClassLabelsList.sorted()),
            Set(expectedClassLabels),
            "検出されたクラスラベル「\(result.metadata.detectedClassLabelsList.sorted())」が期待されるラベル「\(expectedClassLabels)」と一致しません"
        )

        result.saveLog(modelAuthor: authorName, modelName: testModelName, modelVersion: testModelVersion)
        let modelFileDir = URL(fileURLWithPath: result.modelOutputPath).deletingLastPathComponent()
        let resultDir = modelFileDir.appendingPathComponent("MultiClass_Result_1")
        let expectedLogFileName = "MultiClass_Run_Report_\(testModelVersion).md"
        let expectedLogFilePath = resultDir.appendingPathComponent(expectedLogFileName).path
        XCTAssertTrue(
            fileManager.fileExists(atPath: expectedLogFilePath),
            "ログファイルが期待されるパス「\(expectedLogFilePath)」に生成されていません"
        )

        XCTAssertEqual(
            result.metadata.modelName,
            testModelName,
            "訓練結果のmodelName「\(result.metadata.modelName)」が期待値「\(testModelName)」と一致しません"
        )

        do {
            let logContents = try String(contentsOfFile: expectedLogFilePath, encoding: .utf8)
            XCTAssertFalse(logContents.isEmpty, "ログファイルが空です: \(expectedLogFilePath)")
        } catch {
            XCTFail("ログファイルの読み込みに失敗しました: \(expectedLogFilePath), エラー: \(error.localizedDescription)")
        }

        // モデルファイル名の検証
        let modelFilePath = result.metadata.trainedModelFilePath
        let modelFileName = URL(fileURLWithPath: modelFilePath).lastPathComponent
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

        let vnCoreMLModel = try VNCoreMLModel(for: mlModel)
        let predictionRequest = VNCoreMLRequest(model: vnCoreMLModel)

        let classLabelsForPredictionTest = result.metadata.detectedClassLabelsList.sorted()

        guard !classLabelsForPredictionTest.isEmpty else {
            XCTFail("予測テストの実行には、訓練結果に最低1つのクラスラベルが必要です。検出されたラベルはありません。")
            throw ClassifierTestsError.setupFailed
        }

        let classLabelToTest = classLabelsForPredictionTest[0]
        let imageURL: URL
        do {
            imageURL = try TestUtils.getRandomImageURL(
                forClassLabel: classLabelToTest,
                resourcesDirectoryPath: classifier.resourcesDirectoryPath,
                fileManager: fileManager
            )
        } catch {
            XCTFail("クラス「\(classLabelToTest)」のサブディレクトリからのランダム画像取得に失敗しました。エラー: \(error.localizedDescription)")
            throw error
        }

        let imageHandler = VNImageRequestHandler(url: imageURL, options: [:])
        try imageHandler.perform([predictionRequest])
        guard let observations = predictionRequest.results as? [VNClassificationObservation],
              let topResult = observations.first
        else {
            XCTFail("クラス「\(classLabelToTest)」の画像に対する分類結果オブジェクトの取得に失敗しました。")
            throw ClassifierTestsError.predictionFailed
        }
        XCTAssertNotNil(topResult.identifier, "クラス「\(classLabelToTest)」の画像に対する予測結果からクラスラベルが取得できませんでした。")
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
            of: "MultiClass_Result_",
            with: ""
        )) ?? 0
        let secondResultNumber = Int(secondModelFileDir.lastPathComponent.replacingOccurrences(
            of: "MultiClass_Result_",
            with: ""
        )) ?? 0
        XCTAssertEqual(
            secondResultNumber,
            firstResultNumber + 1,
            "2回目の出力ディレクトリの連番が期待値と一致しません。\n1回目: \(firstResultNumber)\n2回目: \(secondResultNumber)"
        )
    }
}

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
            .appendingPathComponent("Binary")
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

    func testClassifierDIConfiguration() async throws {
        // モデルの作成
        let result = try await classifier.create(
            author: authorName,
            modelName: testModelName,
            version: testModelVersion,
            modelParameters: modelParameters
        )
        
        XCTAssertNotNil(classifier, "BinaryClassifierの初期化失敗")
        XCTAssertEqual(classifier.outputParentDirPath, temporaryOutputDirectoryURL.path, "分類器の出力パスが期待値と不一致")
    }

    // モデルの訓練と成果物の生成をテスト
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

        // Binary分類器の期待されるクラスラベル
        let expectedClassLabels = ["not_scary", "scary"].sorted()
        XCTAssertEqual(
            Set(result.metadata.detectedClassLabelsList.sorted()),
            Set(expectedClassLabels),
            "検出されたクラスラベル「\(result.metadata.detectedClassLabelsList.sorted())」が期待されるラベル「\(expectedClassLabels)」と一致しません"
        )

        result.saveLog(modelAuthor: authorName, modelName: testModelName, modelVersion: testModelVersion)
        let modelFileDir = URL(fileURLWithPath: result.metadata.trainedModelFilePath).deletingLastPathComponent()

        // Binary分類器の期待されるログファイル名
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

        // Binary分類器の期待されるモデルファイル名パターン
        let modelFilePath = result.metadata.trainedModelFilePath
        let modelFileName = URL(fileURLWithPath: modelFilePath).lastPathComponent
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
            result.metadata.trainedModelFilePath.contains(testModelVersion),
            "モデルファイルパスにバージョン「\(testModelVersion)」が含まれていません"
        )
        XCTAssertTrue(
            result.metadata.trainedModelFilePath.contains("Binary"),
            "モデルファイルパスに分類法「Binary」が含まれていません"
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

        let classLabels = result.metadata.detectedClassLabelsList.sorted()
        guard classLabels.count >= 2 else {
            XCTFail("テストには少なくとも2つのクラスラベルが訓練結果に必要です。検出されたラベル: \(classLabels)")
            throw ClassifierTestsError.setupFailed
        }

        let classLabel1 = classLabels[0]
        let classLabel2 = classLabels[1]

        print("動的に識別されたテスト用クラスラベル: '\(classLabel1)' および '\(classLabel2)'")

        // classLabel1 の画像取得処理
        let imageURL1: URL
        do {
            imageURL1 = try TestUtils.getRandomImageURL(
                forClassLabel: classLabel1,
                resourcesDirectoryPath: classifier.resourcesDirectoryPath,
                fileManager: fileManager
            )
        } catch {
            XCTFail("'\(classLabel1)' サブディレクトリからのランダム画像取得失敗。エラー: \(error.localizedDescription)")
            throw error
        }

        // classLabel2 の画像取得処理
        let imageURL2: URL
        do {
            imageURL2 = try TestUtils.getRandomImageURL(
                forClassLabel: classLabel2,
                resourcesDirectoryPath: classifier.resourcesDirectoryPath,
                fileManager: fileManager
            )
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
            throw ClassifierTestsError.predictionFailed
        }
        XCTAssertNotNil(topResult1.identifier, "クラス '\(classLabel1)' 画像: 予測結果からクラスラベルを取得できませんでした。")
        print("クラス '\(classLabel1)' 画像の予測 (正解ラベル): \(topResult1.identifier) (確信度: \(topResult1.confidence))")

        let imageHandler2 = VNImageRequestHandler(url: imageURL2, options: [:])
        try imageHandler2.perform([predictionRequest])
        guard let observations2 = predictionRequest.results as? [VNClassificationObservation],
              let topResult2 = observations2.first
        else {
            XCTFail("クラス '\(classLabel2)' 画像: 有効な分類結果オブジェクトを取得できませんでした。")
            throw ClassifierTestsError.predictionFailed
        }
        XCTAssertNotNil(topResult2.identifier, "クラス '\(classLabel2)' 画像: 予測結果からクラスラベルを取得できませんでした。")
        print("クラス '\(classLabel2)' 画像の予測 (正解ラベル): \(topResult2.identifier) (確信度: \(topResult2.confidence))")
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
            of: "Binary_Result_",
            with: ""
        )) ?? 0
        let secondResultNumber = Int(secondModelFileDir.lastPathComponent.replacingOccurrences(
            of: "Binary_Result_",
            with: ""
        )) ?? 0
        XCTAssertEqual(
            secondResultNumber,
            firstResultNumber + 1,
            "2回目の出力ディレクトリの連番が期待値と一致しません。\n1回目: \(firstResultNumber)\n2回目: \(secondResultNumber)"
        )
    }

}

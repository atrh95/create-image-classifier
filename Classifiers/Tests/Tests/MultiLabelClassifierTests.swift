import CICFileManager
import CoreML
import CreateML
import Foundation
import MultiLabelClassifier
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
            .appendingPathComponent("MultiLabelResources")
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
        try await classifier.create(
            author: authorName,
            modelName: testModelName,
            version: testModelVersion,
            modelParameters: modelParameters
        )

        XCTAssertNotNil(classifier, "MultiLabelClassifierの初期化失敗")
        XCTAssertEqual(classifier.outputParentDirPath, temporaryOutputDirectoryURL.path, "分類器の出力パスが期待値と不一致")
    }

    // モデルの訓練と成果物の生成をテスト
    func testModelTrainingAndArtifactGeneration() async throws {
        // モデルの作成
        try await classifier.create(
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

        // クラスラベルディレクトリの取得
        let classLabelDirs = try fileManager.contentsOfDirectory(
            at: URL(fileURLWithPath: classifier.resourcesDirectoryPath),
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ).filter(\.hasDirectoryPath)
            .map(\.lastPathComponent)
            .sorted()

        // クラスラベルの存在確認
        XCTAssertFalse(
            classLabelDirs.isEmpty,
            "リソースディレクトリにクラスラベルディレクトリが見つかりません: \(classifier.resourcesDirectoryPath)"
        )

        // 各クラスラベルディレクトリにファイルが存在することを確認
        for classLabel in classLabelDirs {
            let classDirURL = URL(fileURLWithPath: classifier.resourcesDirectoryPath)
                .appendingPathComponent(classLabel)
            let files = try fileManager.contentsOfDirectory(
                at: classDirURL,
                includingPropertiesForKeys: nil
            )

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
            modelFilePath.contains(testModelVersion),
            "モデルファイルパスにバージョン「\(testModelVersion)」が含まれていません"
        )
        XCTAssertTrue(
            modelFilePath.contains("MultiLabel"),
            "モデルファイルパスに分類法「MultiLabel」が含まれていません"
        )
    }

    // モデルが予測を実行できるかテスト
    func testModelCanPerformPrediction() async throws {
        // モデルの作成
        try await classifier.create(
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

        print("結果ディレクトリ: \(latestResultDir.path)")

        // モデルファイルの存在確認
        let modelFiles = try fileManager.contentsOfDirectory(
            at: latestResultDir,
            includingPropertiesForKeys: nil
        ).filter { $0.pathExtension == "mlmodel" }

        print("検出されたモデルファイル: \(modelFiles.map(\.lastPathComponent).joined(separator: ", "))")

        guard let modelFile = modelFiles.first else {
            XCTFail("モデルファイルが見つかりません: \(latestResultDir.path)")
            return
        }

        // モデルのコンパイル
        let compiledModelURL = try await MLModel.compileModel(at: modelFile)
        print("コンパイルされたモデル: \(compiledModelURL.path)")

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

        let classLabels = try fileManager.contentsOfDirectory(
            at: URL(fileURLWithPath: classifier.resourcesDirectoryPath),
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ).filter(\.hasDirectoryPath)
            .map(\.lastPathComponent)
            .sorted()

        guard classLabels.count >= 2 else {
            XCTFail("テストには少なくとも2つのクラスラベルが必要です。検出されたラベル: \(classLabels)")
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
        try await classifier.create(
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

        guard let firstResultDir = firstResultDirs.first else {
            XCTFail("1回目の結果ディレクトリが見つかりません: \(outputDir.path)")
            return
        }

        // 2回目のモデル作成を実行
        try await classifier.create(
            author: "TestAuthor",
            modelName: testModelName,
            version: "v1",
            modelParameters: modelParameters
        )

        // 出力ディレクトリから最新の結果を取得
        let secondResultDirs = try fileManager.contentsOfDirectory(
            at: outputDir,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ).filter(\.hasDirectoryPath)
            .sorted { $0.lastPathComponent > $1.lastPathComponent }

        guard let secondResultDir = secondResultDirs.first else {
            XCTFail("2回目の結果ディレクトリが見つかりません: \(outputDir.path)")
            return
        }

        // 連番の検証
        let firstResultNumber = Int(firstResultDir.lastPathComponent.replacingOccurrences(
            of: "MultiLabel_Result_",
            with: ""
        )) ?? 0
        let secondResultNumber = Int(secondResultDir.lastPathComponent.replacingOccurrences(
            of: "MultiLabel_Result_",
            with: ""
        )) ?? 0

        print("1回目の結果ディレクトリ: \(firstResultDir.lastPathComponent)")
        print("2回目の結果ディレクトリ: \(secondResultDir.lastPathComponent)")
        print("1回目の番号: \(firstResultNumber)")
        print("2回目の番号: \(secondResultNumber)")

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

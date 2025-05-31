import CICFileManager
import CoreML
import CreateML
import Foundation
@testable import OvRClassification
import Vision
import XCTest

final class OvRClassifierTests: XCTestCase {
    var classifier: OvRClassifier!
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

        let resourceDirectoryPath = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // OvRClassifierTests
            .appendingPathComponent("TestResources")
            .appendingPathComponent("OvR")
            .path

        classifier = OvRClassifier(
            outputDirectoryPathOverride: temporaryOutputDirectoryURL.path,
            resourceDirPathOverride: resourceDirectoryPath
        )

        // モデルの作成
        trainingResult = await classifier.create(
            author: authorName,
            modelName: testModelName,
            version: testModelVersion,
            modelParameters: modelParameters,
            scenePrintRevision: nil
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
        classifier.resourceDirPathOverride = nil
        classifier = nil
        try super.tearDownWithError()
    }

    func testClassifierDIConfiguration() throws {
        XCTAssertNotNil(classifier, "OvRClassifierの初期化失敗")
        XCTAssertEqual(classifier.outputParentDirPath, temporaryOutputDirectoryURL.path, "分類器の出力パスが期待値と不一致")
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

        // モデルファイル名の検証
        let modelFilePath = result.metadata.trainedModelFilePath
        let modelFileName = URL(fileURLWithPath: modelFilePath).lastPathComponent
        
        // OvR分類器では各クラスに対して1つの分類器を作成するため、各クラス名を含むパターンを期待
        let regex = #"^TestModel_OvR_[a-z_]+_v\d+\.mlmodel$"#
        XCTAssertTrue(modelFileName.range(of: regex, options: .regularExpression) != nil,
                     """
                     モデルファイル名が期待パターンに一致しません。
                     期待パターン: \(regex)
                     実際: \(modelFileName)
                     """)

        XCTAssertTrue(
            result.modelOutputPath.contains(testModelVersion),
            "モデルファイルのパスにバージョン「\(testModelVersion)」が含まれていません"
        )
        XCTAssertTrue(
            result.modelOutputPath.contains(classifier.classificationMethod),
            "モデルファイルのパスに分類手法「\(classifier.classificationMethod)」が含まれていません"
        )

        result.saveLog(modelAuthor: authorName, modelName: testModelName, modelVersion: testModelVersion)
        let modelFileDir = URL(fileURLWithPath: result.metadata.trainedModelFilePath).deletingLastPathComponent()
        let expectedLogFileName = "OvR_Run_Report_\(testModelVersion).md"
        let expectedLogFilePath = modelFileDir.appendingPathComponent(expectedLogFileName).path
        XCTAssertTrue(fileManager.fileExists(atPath: expectedLogFilePath), "ログファイル「\(expectedLogFilePath)」が生成されていません")

        XCTAssertEqual(result.metadata.modelName, testModelName)
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

    /// 各クラスにおいて、一時ディレクトリ内の正例クラスとrestクラスの画像枚数が適切にバランスされていることを検証する
    func testClassImageBalance() throws {
        guard let result = trainingResult else {
            XCTFail("訓練結果がnilです")
            throw TestError.trainingFailed
        }

        // 一時ディレクトリのパスを取得
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(OvRClassifier.tempBaseDirName)
        
        // 一時ディレクトリが存在することを確認
        XCTAssertTrue(FileManager.default.fileExists(atPath: tempDir.path), "一時ディレクトリが存在しません: \(tempDir.path)")
        
        // 一時ディレクトリ内のクラスディレクトリを取得
        let classDirs = try FileManager.default.contentsOfDirectory(at: tempDir, includingPropertiesForKeys: [.isDirectoryKey])
            .filter { $0.hasDirectoryPath }
        
        // 正例クラスとrestクラスのディレクトリを取得
        let positiveClassDir = classDirs.first { $0.lastPathComponent != "rest" }
        let restDir = classDirs.first { $0.lastPathComponent == "rest" }
        
        XCTAssertNotNil(positiveClassDir, "正例クラスのディレクトリが見つかりません")
        XCTAssertNotNil(restDir, "restクラスのディレクトリが見つかりません")
        
        // 正例クラスの画像枚数を確認
        let positiveFiles = try FileManager.default.contentsOfDirectory(at: positiveClassDir!, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension.lowercased() == "jpg" || $0.pathExtension.lowercased() == "jpeg" || $0.pathExtension.lowercased() == "png" }
        
        // restクラスの画像枚数を確認
        let restFiles = try FileManager.default.contentsOfDirectory(at: restDir!, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension.lowercased() == "jpg" || $0.pathExtension.lowercased() == "jpeg" || $0.pathExtension.lowercased() == "png" }
        
        // 正例クラスの枚数を取得
        let positiveCount = positiveFiles.count
        
        // restクラス数（正例クラスを除く）
        let restClassCount = try FileManager.default.contentsOfDirectory(at: URL(fileURLWithPath: classifier.resourcesDirectoryPath), includingPropertiesForKeys: [.isDirectoryKey])
            .filter { $0.hasDirectoryPath }
            .count - 1
        
        // 各restクラスから取得されるべき枚数（切り上げ除算）
        let expectedSamplesPerRestClass = Int(ceil(Double(positiveCount) / Double(restClassCount)))
        
        // restの合計枚数が期待値と一致することを確認
        let expectedTotalRestCount = expectedSamplesPerRestClass * restClassCount
        XCTAssertEqual(restFiles.count, expectedTotalRestCount,
                      """
                      restクラスの合計枚数が期待値と一致しません。
                      正例クラス [\(positiveClassDir!.lastPathComponent)]: \(positiveCount)枚
                      restクラス数: \(restClassCount)
                      期待される各restクラスの枚数: \(expectedSamplesPerRestClass)
                      期待される合計rest枚数: \(expectedTotalRestCount)
                      実際のrest枚数: \(restFiles.count)
                      """)
        
        // 正例クラスの枚数が0でないことを確認
        XCTAssertGreaterThan(positiveCount, 0, "正例クラスの枚数が0です")
        
        // restクラスの枚数が0でないことを確認
        XCTAssertGreaterThan(restFiles.count, 0, "restクラスの枚数が0です")
    }

    /// 各クラスにおいて、一時ディレクトリ内の正例クラスと負例クラスの画像枚数が一致していることを検証する
    func testBinaryClassImageBalance() throws {
        let resourceURL = URL(fileURLWithPath: classifier.resourcesDirectoryPath)
        let classLabels = try FileManager.default.contentsOfDirectory(at: resourceURL, includingPropertiesForKeys: [.isDirectoryKey])
            .filter { $0.hasDirectoryPath }
            .map { $0.lastPathComponent }
        
        // 各クラスで画像枚数のバランスを確認
        for positiveClass in classLabels {
            // トレーニングデータを準備
            _ = try classifier.prepareTrainingData(positiveClass: positiveClass, basePath: resourceURL.path)
            
            // クラス間の画像枚数を取得
            let (positiveCount, negativeCount) = try classifier.balanceClassImages(positiveClass: positiveClass, basePath: resourceURL.path)
            
            // 一時ディレクトリのパスを取得
            let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(OvRClassifier.tempBaseDirName)
            let tempPositiveDir = tempDir.appendingPathComponent(positiveClass)
            let tempNegativeDir = tempDir.appendingPathComponent("negative")
            
            // 一時ディレクトリ内の画像枚数を確認
            let tempPositiveFiles = try FileManager.default.contentsOfDirectory(at: tempPositiveDir, includingPropertiesForKeys: nil)
                .filter { $0.pathExtension.lowercased() == "jpg" || $0.pathExtension.lowercased() == "jpeg" || $0.pathExtension.lowercased() == "png" }
            
            let tempNegativeFiles = try FileManager.default.contentsOfDirectory(at: tempNegativeDir, includingPropertiesForKeys: nil)
                .filter { $0.pathExtension.lowercased() == "jpg" || $0.pathExtension.lowercased() == "jpeg" || $0.pathExtension.lowercased() == "png" }
            
            // 画像枚数が一致することを確認
            XCTAssertEqual(tempPositiveFiles.count, positiveCount, "正例クラス [\(positiveClass)] の画像枚数が一致しません。期待値: \(positiveCount), 実際: \(tempPositiveFiles.count)")
            XCTAssertEqual(tempNegativeFiles.count, negativeCount, "負例クラスの画像枚数が一致しません。期待値: \(negativeCount), 実際: \(tempNegativeFiles.count)")
            XCTAssertEqual(tempPositiveFiles.count, tempNegativeFiles.count, "正例クラス [\(positiveClass)] と負例クラスの画像枚数が一致しません。")
        }
    }

    // 出力ディレクトリの連番を検証
    func testSequentialOutputDirectoryNumbering() async throws {
        // 1回目のモデル作成（setUpで実行済み）
        guard let firstResult = trainingResult else {
            XCTFail("1回目の訓練結果がnilです")
            throw TestError.trainingFailed
        }

        let firstModelFileDir = URL(fileURLWithPath: firstResult.metadata.trainedModelFilePath).deletingLastPathComponent()

        // 2回目のモデル作成を実行
        let secondResult = await classifier.create(
            author: "TestAuthor",
            modelName: testModelName,
            version: "v1",
            modelParameters: modelParameters,
            scenePrintRevision: nil
        )

        guard let secondResult = secondResult else {
            XCTFail("2回目の訓練結果がnilです")
            throw TestError.trainingFailed
        }

        let secondModelFileDir = URL(fileURLWithPath: secondResult.metadata.trainedModelFilePath).deletingLastPathComponent()
        
        // 連番の検証
        let firstResultNumber = Int(firstModelFileDir.lastPathComponent.replacingOccurrences(of: "OvR_Result_", with: "")) ?? 0
        let secondResultNumber = Int(secondModelFileDir.lastPathComponent.replacingOccurrences(of: "OvR_Result_", with: "")) ?? 0
        XCTAssertEqual(
            secondResultNumber,
            firstResultNumber + 1,
            "2回目の出力ディレクトリの連番が期待値と一致しません。\n1回目: \(firstResultNumber)\n2回目: \(secondResultNumber)"
        )
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

        guard let randomFile = allFiles.randomElement() else {
            XCTFail("利用可能な画像ファイルが見つかりません")
            throw TestError.testResourceMissing
        }

        return randomFile
    }
}

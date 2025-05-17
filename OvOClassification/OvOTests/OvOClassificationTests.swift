@testable import OvOClassification // テスト対象モジュールをインポート
import XCTest
import CoreML
import CreateML // MLImageClassifier.ModelParameters のために必要 (OvOでも使うか要確認)
import Vision // VNCoreMLModel, VNImageRequestHandler, VNClassificationObservation のために必要
import Foundation

class OvOClassificationTests: XCTestCase {

    var trainer: OvOClassificationTrainer! // OvOトレーナーに変更
    let fileManager = FileManager.default
    let authorName: String = "Test Author"
    let testModelName: String = "TestCats_OvO_Run"
    let testModelVersion: String = "v1"

    let algorithm = MLImageClassifier.ModelParameters.ModelAlgorithmType.transferLearning(
        featureExtractor: .scenePrint(revision: 1),
        classifier: .logisticRegressor
    )

    var modelParameters: MLImageClassifier.ModelParameters {
        MLImageClassifier.ModelParameters(
            validation: .split(strategy: .automatic),
            maxIterations: 10,
            augmentation: [],
            algorithm: algorithm
        )
    }
    
    // テストリソースへのパス
    var testResourcesRootPath: String {
        var currentTestFileDir = URL(fileURLWithPath: #filePath)
        currentTestFileDir.deleteLastPathComponent()
        return currentTestFileDir.appendingPathComponent("TestResources").path
    }

    // OvO用のアノテーションファイル名またはデータ構造に応じて調整が必要
    // 例: JSONファイルを使用する場合
    private func findAnnotationFileName() -> String? {
        // MultiLabelと同様のJSON探索ロジックを仮定。OvOの仕様に合わせて変更。
        do {
            let resourceURL = URL(fileURLWithPath: testResourcesRootPath)
            let items = try fileManager.contentsOfDirectory(at: resourceURL, includingPropertiesForKeys: nil)
            // OvO用のアノテーションファイル形式に応じて拡張子やファイル名パターンを変更
            if let jsonFile = items.first(where: { $0.pathExtension.lowercased() == "json" && $0.lastPathComponent.contains("ovo_annotations") }) {
                return jsonFile.lastPathComponent
            } else {
                XCTFail("OvOテストリソースディレクトリに適切なJSONアノテーションファイルが見つかりません: \(testResourcesRootPath)")
                return nil
            }
        } catch {
            XCTFail("OvOテストリソースディレクトリの読み取りに失敗しました: \(testResourcesRootPath) - \(error.localizedDescription)")
            return nil
        }
    }

    var resolvedAnnotationFileName: String! 

    var temporaryOutputDirectoryURL: URL!
    var compiledModelURL: URL? // OvOの場合、複数のモデルが生成される可能性も考慮
    var trainingResult: OvOClassification.OvOTrainingResult? // OvOの訓練結果型に変更

    enum TestError: Error {
        case trainingFailed
        case modelFileMissing // 単一または複数のモデル
        case predictionFailed
        case setupFailed
        case resourcePathError
        case annotationFileError // OvO用のアノテーションエラー
    }

    override func setUp() async throws {
        try await super.setUp()
        
        resolvedAnnotationFileName = findAnnotationFileName()
        guard resolvedAnnotationFileName != nil else {
            throw TestError.annotationFileError
        }

        temporaryOutputDirectoryURL = fileManager.temporaryDirectory
            .appendingPathComponent("TestOutput_OvO_\(UUID().uuidString)")
        try fileManager.createDirectory(
            at: temporaryOutputDirectoryURL,
            withIntermediateDirectories: true,
            attributes: nil
        )

        trainer = OvOClassificationTrainer(
            resourcesDirectoryPathOverride: testResourcesRootPath,
            outputDirectoryPathOverride: temporaryOutputDirectoryURL.path
        )
        
        trainingResult = await trainer.train(
            author: authorName,
            modelName: testModelName,
            version: testModelVersion,
            modelParameters: self.modelParameters 
        )

        guard let result = trainingResult else {
            throw TestError.trainingFailed
        }

        // OvOの場合、メインのモデルまたは代表モデルのパスを取得・コンパイル
        // 複数のバイナリ分類器が生成される場合、それらをどう扱うかテスト方針による
        let mainModelPath = result.modelOutputPath // OvOTrainingResult に適切なパスがある前提
        let trainedModelURL = URL(fileURLWithPath: mainModelPath)
        
        // 注意: OvOで単一の .mlmodel が出力されるとは限らない。
        // CreateMLの OvO 実装がどうなっているか、または自前の OvO 実装詳細による。
        // ここでは仮に単一モデルとしてコンパイル。
        do {
            compiledModelURL = try await MLModel.compileModel(at: trainedModelURL)
        } catch {
            print("モデルのコンパイル失敗 (OvO setUp): \(error.localizedDescription)")
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
        // OvOで複数のモデルファイルを扱う場合は、それらも削除
        compiledModelURL = nil
        trainingResult = nil
        trainer = nil
        try super.tearDownWithError()
    }

    // MARK: - Test Cases

    func testTrainerInitialization() throws {
        XCTAssertNotNil(trainer, "OvOClassificationTrainerの初期化失敗")
        XCTAssertEqual(trainer.resourcesDirectoryPath, testResourcesRootPath, "トレーナーのリソースパスが期待値と不一致")
        XCTAssertEqual(trainer.outputDirPath, temporaryOutputDirectoryURL.path, "トレーナーの出力パスが期待値と不一致")
        // OvO特有の初期化パラメータがあれば追加でテスト
    }

    func testModelTrainingAndArtifactGeneration() throws {
        guard let result = trainingResult else {
            XCTFail("訓練結果がnilです (OvO testModelTrainingAndArtifactGeneration)")
            throw TestError.trainingFailed
        }

        // OvOの場合、成果物は複数になる可能性が高い (例: 各ペアごとのモデル)
        // ここでは代表的なモデルパスの存在を確認
        XCTAssertTrue(
            fileManager.fileExists(atPath: result.modelOutputPath),
            "訓練モデルファイル(メイン)が期待されるパス「\(result.modelOutputPath)」に見つかりません"
        )

        // OvOのクラスラベルの期待値を設定 (例: ["classA", "classB", "classC"])
        // result.classLabels の持ち方によって調整
        let expectedClassLabels = ["cat", "dog", "rabbit"].sorted() // 仮のラベル
        // XCTAssertEqual(Set(result.classLabels.sorted()), Set(expectedClassLabels), "訓練結果のクラスラベルが期待値と不一致")

        // ログファイルの生成確認 (OvOの仕様に合わせて)
        result.saveLog(modelAuthor: authorName, modelName: testModelName, modelVersion: testModelVersion)
        let modelFileDir = URL(fileURLWithPath: result.modelOutputPath).deletingLastPathComponent()
        let expectedLogFileName = "\(testModelName)_\(testModelVersion).md"
        let expectedLogFilePath = modelFileDir.appendingPathComponent(expectedLogFileName).path
        XCTAssertTrue(fileManager.fileExists(atPath: expectedLogFilePath), "ログファイル「\(expectedLogFilePath)」が生成されていません")

        // OvOTrainingResult特有のフィールドの検証
        // 例: XCTAssertNotNil(result.binaryClassifierResults, "バイナリ分類器群の結果がnilです")
    }

    func testModelCanPerformPrediction() throws {
        guard let finalModelURL = compiledModelURL else { // OvOの場合、予測に使うモデル(群)のURL
            XCTFail("コンパイル済みモデルのURLがnil (OvO testModelCanPerformPrediction)")
            throw TestError.modelFileMissing
        }
        guard trainingResult != nil else {
            XCTFail("訓練結果がnil (OvO testModelCanPerformPrediction)")
            throw TestError.trainingFailed
        }

        // OvOの予測テスト用画像データの準備
        // 例えば、いずれかのクラスに属するテスト画像を取得
        // アノテーションファイルまたはディレクトリ構造からテスト画像を選択
        let testImageName = "test_cat_for_ovo.jpg" // 仮の画像ファイル名、実際のテストリソースに合わせてください
        let imageURL = URL(fileURLWithPath: testResourcesRootPath).appendingPathComponent(testImageName)

        guard fileManager.fileExists(atPath: imageURL.path) else {
            XCTFail("OvOテスト用画像ファイルが見つかりません: \(imageURL.path)")
            throw TestError.resourcePathError
        }

        // OvOの予測ロジック
        // 1. もし単一の .mlmodel でOvO全体がラップされている場合:
        let mlModel = try MLModel(contentsOf: finalModelURL)
        let visionModel = try VNCoreMLModel(for: mlModel)
        let request = VNCoreMLRequest(model: visionModel) { request, error in
            if let error = error {
                XCTFail("VNCoreMLRequest failed: \(error.localizedDescription)")
                return
            }
            guard let observations = request.results as? [VNClassificationObservation] else {
                XCTFail("予測結果をVNClassificationObservationにキャストできませんでした。")
                return
            }
            
            // 目的: 何らかの予測が行われたことを確認する（正しさは問わない）
            XCTAssertFalse(observations.isEmpty, "予測結果(observations)が空でした。モデルは予測を行いませんでした。")

            if let topResult = observations.first {
                print("OvO Top prediction for \(imageURL.lastPathComponent): \(topResult.identifier) with confidence \(topResult.confidence)")
            } else {
                // このケースは XCTAssertFalse(observations.isEmpty, ...) によってカバーされる
                print("OvO prediction for \(imageURL.lastPathComponent): No observations found, though this should have been caught by XCTAssertFalse.")
            }
        }
        
        let handler = VNImageRequestHandler(url: imageURL, options: [:])
        try handler.perform([request])

        // 2. 複数のバイナリ分類器モデルを組み合わせて予測する場合 (より複雑):
        //    - 各バイナリ分類器で予測を実行
        //    - 投票ロジックを適用
        //    - 最終結果を検証
        //    このパターンのテストは、OvOClassificationTrainerの実装に大きく依存するため、
        //    ここでは上記単一モデルパターンのみを実装しています。
        //    必要に応じて、このセクションのコメントアウトを解除し、実装してください。
    }
}

// OvOClassificationTrainer.ManifestEntry や OvOTrainingResult など、
// OvOClassificationTrainer.swift で定義されていてテストで必要な型があれば、
// MultiLabelと同様に extension で再定義するか、アクセスレベルを調整する必要があるかもしれません。
// 例:
// fileprivate extension OvOClassificationTrainer {
//     struct OvOAnnotationEntry: Decodable { ... }
// }

// public struct OvOTrainingResult { ... } のように OvOClassificationTrainer.swift 内で定義されている場合、
// OvOClassification モジュールを import していれば直接使えるはず。
// そうでない場合は、テスト内で必要な部分をスタブとして定義するか、アクセスレベルを変更。

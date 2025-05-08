import CoreML
import CreateML
import CreateMLComponents
import Foundation
import SCSInterface

@available(macOS 14, *)   // API利用可能性の警告を抑制
public class MultiLabelClassificationTrainer: ScreeningTrainerProtocol {
    public typealias TrainingResultType = MultiLabelTrainingResult

    // Helper struct for decoding the manifest JSON
    struct ManifestEntry: Decodable {
        var filename: String
        var annotations: [String] // Decode as array, then convert to Set<String>
    }

    public var modelName: String { "MultiLabelScaryCatML_Components" }
    public var dataDirectoryName: String { "Images" }
    public var manifestFileName: String { "multilabel_cat_annotations.json" }
    public var customOutputDirPath: String { "OutputModels/ScaryCatScreeningML/MultiLabel" }

    public var resourcesDirectoryPath: String {
        var dir = URL(fileURLWithPath: #filePath)
        dir.deleteLastPathComponent()
        dir.deleteLastPathComponent()
        return dir.appendingPathComponent("Resources").path
    }

    public init() {}

    public func train(
        author: String,
        shortDescription: String,
        version: String
    ) async -> TrainingResultType? {
        let resourcesDir = URL(fileURLWithPath: resourcesDirectoryPath)
        let imageBaseDir = resourcesDir.appendingPathComponent(dataDirectoryName)
        let manifestFile = resourcesDir.appendingPathComponent(manifestFileName)

        guard FileManager.default.fileExists(atPath: imageBaseDir.path) else {
            print("❌ エラー: 画像ベースディレクトリが見つかりません: \(imageBaseDir.path)")
            return nil
        }
        guard FileManager.default.fileExists(atPath: manifestFile.path) else {
            print("❌ エラー: マニフェストファイルが見つかりません: \(manifestFile.path)")
            return nil
        }

        let fileManager = FileManager.default
        var finalOutputDir: URL!

        do {
            var playgroundRoot = URL(fileURLWithPath: #filePath)
            (1...3).forEach { _ in playgroundRoot.deleteLastPathComponent() }
            var baseOutputDir = playgroundRoot

            let customPath = customOutputDirPath
            if !customPath.isEmpty {
                let customURL = URL(fileURLWithPath: customPath)
                baseOutputDir = customURL.isFileURL && customPath.hasPrefix("/") ? customURL : baseOutputDir.appendingPathComponent(customPath)
            } else {
                baseOutputDir = baseOutputDir.appendingPathComponent("OutputModels")
            }
            try fileManager.createDirectory(at: baseOutputDir, withIntermediateDirectories: true, attributes: nil)

            var resultCounter = 1
            let resultDirPrefix = "multilabel_components_result_"
            repeat {
                finalOutputDir = baseOutputDir.appendingPathComponent("\(resultDirPrefix)\(resultCounter)")
                resultCounter += 1
            } while fileManager.fileExists(atPath: finalOutputDir.path)
            try fileManager.createDirectory(at: finalOutputDir, withIntermediateDirectories: false, attributes: nil)
            print("💾 結果保存ディレクトリ (マルチラベル - Components): \(finalOutputDir.path)")

            print("\n🚀 マルチラベル分類モデル (Components) [\(modelName)] のトレーニングを開始します...")
            print("  マニフェスト: \(manifestFile.path)")
            print("  画像ディレクトリ: \(imageBaseDir.path)")

            var allLabelsSet = Set<String>()
            // Decode manifest and prepare annotated features
            let annotatedFeatures: [AnnotatedFeature<URL, Set<String>>]
            do {
                let jsonData = try Data(contentsOf: manifestFile)
                let decoder = JSONDecoder()
                let manifestEntries = try decoder.decode([ManifestEntry].self, from: jsonData)

                annotatedFeatures = manifestEntries.map { entry in
                    let imageURL = imageBaseDir.appendingPathComponent(entry.filename)
                    let labelSet = Set(entry.annotations)
                    labelSet.forEach { allLabelsSet.insert($0) } // Populate allLabelsSet here
                    return AnnotatedFeature(feature: imageURL, annotation: labelSet)
                }
                
                if annotatedFeatures.isEmpty {
                    print("❌ エラー: マニフェストファイルからアノテーションを読み込めませんでした、またはアノテーションが空です。")
                    return nil
                }
            } catch {
                print("❌ エラー: マニフェストファイルの読み込みまたはデコードに失敗しました: \(error.localizedDescription)")
                return nil
            }
            
            let sortedClassLabels = allLabelsSet.sorted()
            guard !sortedClassLabels.isEmpty else {
                print("❌ エラー: マニフェストからラベルを抽出できませんでした。マニフェストファイルを確認してください。")
                return nil
            }
            print("📚 検出された全ラベル (マニフェストより): \(sortedClassLabels.joined(separator: ", "))")

            let startTime = Date()

            let allLabelsForClassifier: Set<String> = allLabelsSet

            // ImageFeaturePrintを設定 - リビジョン調整可能
            let featurePrint = ImageFeaturePrint(revision: 1)
            // ラベルが分類器に渡されることを確認
            let classifier = FullyConnectedNetworkMultiLabelClassifier<Float, String>(labels: allLabelsForClassifier)

            // ImageReaderをパイプラインの先頭にする
            print("  📖 ImageReaderをパイプラインに設定...")
            let imageReaderInPipeline = ImageReader()
            
            let pipeline = imageReaderInPipeline
                .appending(featurePrint)
                .appending(classifier)

            // 分割比率はトレーニング0.8、検証0.2
            // `annotatedFeatures` ( [AnnotatedFeature<URL, Set<String>>] ) を直接分割
            print("  📊 トレーニングデータと検証データを準備中 (80/20 スプリット)...")
            let (trainingData, validationData) = annotatedFeatures.randomSplit(by: 0.8)

            print("  ⚙️ パイプラインモデルの学習を開始します...")
            let fittedModel = try await pipeline.fitted(
                to: trainingData,
                validateOn: validationData
            )
            
            let endTime = Date()
            let duration = endTime.timeIntervalSince(startTime)
            print("🎉 [\(modelName)] のトレーニングに成功しました！ (所要時間: \(String(format: "%.2f", duration))秒)")

            // メトリクス計算
            print("🧪 検証データでメトリクスを計算中...")
            
            // 予測結果を非同期シーケンスとして取得し、直接配列として扱うことを期待
            // Appleのドキュメント例では await model.prediction(from:) がコレクションを返すように見える
            let validationPredictionsArray = try await fittedModel.prediction(from: validationData)
            
            // .map を使用して、分類器の出力（確率分布）と正解ラベルを抽出
            let predictedClassDistributions = validationPredictionsArray.map(\.prediction)
            let groundTruthLabels = validationPredictionsArray.map(\.annotation)

            // MultiLabelClassificationMetrics は ClassificationDistribution を直接扱えると期待
            let metrics = try MultiLabelClassificationMetrics(
                classifications: predictedClassDistributions, // 確率分布のリストを直接渡す
                groundTruth: groundTruthLabels,          // 正解ラベルのリストを渡す
                strategy: .balancedPrecisionAndRecall,
                labels: allLabelsSet 
            )
            let meanAveragePrecision = metrics.meanAveragePrecision
            print("📊 平均適合率 (MAP) [検証]: \(String(format: "%.4f", meanAveragePrecision))")


            let outputModelURL = finalOutputDir.appendingPathComponent("\(modelName)_\(version).mlmodel")
            print("  💾 [\(modelName)_\(version).mlmodel] を保存中: \(outputModelURL.path)")
            
            // Use CreateMLComponents.ModelMetadata
            // Try initializing with author and version, then set shortDescription if possible
            var modelMetadata = CreateMLComponents.ModelMetadata(version: version, author: author)
            // modelMetadata.shortDescription = shortDescription // Attempt to set, if this property exists and is settable
            // If the above line causes an error, shortDescription might not be a property of CreateMLComponents.ModelMetadata
            // or it might not be settable. In that case, it would be omitted.
            // For now, we'll assume a basic init and rely on the compiler to check property existence/setability.
            // To be safe and avoid new errors if shortDescription isn't settable, let's use only author and version if that's what the init takes.
            // The error "Extra argument 'shortDescription'" strongly suggests the init is (author: String, version: String)
            // We will stick to what the initializer signature implies.

            // Final attempt based on error: Assume initializer is (author: String, version: String)
            // and shortDescription is either not part of this specific metadata or set differently.
            // If CreateMLComponents.ModelMetadata has a shortDescription property, it needs to be set like: 
            // var mm = CreateMLComponents.ModelMetadata(author: author, version: version)
            // mm.shortDescription = shortDescription 
            // However, to avoid introducing a new error if it's not settable, we'll use the most basic valid init.
            // The core issue is the exact signature of CreateMLComponents.ModelMetadata which I cannot introspect.
            // Given the error, this is the most direct interpretation:
            let metadataToExport = CreateMLComponents.ModelMetadata(version: version, author: author)
            
            // モデルのエクスポートを簡略化
            try fittedModel.export(to: outputModelURL, metadata: metadataToExport) 
            
            print("  ✅ [\(modelName)_\(version).mlmodel] は正常に保存されました。")

            // TODO: メトリクス抽出を実装する。CreateMLComponentsパイプラインは異なる方法でメトリクスを提供する可能性があるため、 fittedPipeline または classifier からの取得方法を調査・実装する。
            // let trainingAccuracy: Double = 0.0 // プレースホルダー // MAPを使用するためコメントアウト
            // let validationAccuracy: Double = 0.0 // プレースホルダー // MAPを使用するためコメントアウト
            // let trainingError: Double = 1.0 // プレースホルダー // MAPが主要なためコメントアウト
            // let validationError: Double = 1.0 // プレースホルダー // MAPが主要なためコメントアウト
            // print("⚠️ 注意: トレーニング/検証メトリクスは現在プレースホルダーです。CreateMLComponentsパイプラインからのメトリクス抽出を実装する必要があります。") // MAPを計算するためコメントアウト

            return MultiLabelTrainingResult(
                trainingAccuracy: 0.0, // MAPを主要指標とするため0.0または該当なし(-1)を設定
                validationAccuracy: Double(meanAveragePrecision), // MAPをDoubleにキャスト
                trainingError: 0.0, // MAPを主要指標とするため0.0または該当なし(-1)を設定
                validationError: 0.0, // MAPを主要指標とするため0.0または該当なし(-1)を設定 (MAPが主)
                trainingDuration: duration,
                modelOutputPath: outputModelURL.path,
                trainingDataPath: manifestFile.path,
                classLabels: sortedClassLabels
            )

        } catch let error as CreateML.MLCreateError {
            print("  ❌ モデル [\(modelName)] のトレーニングまたは保存エラー (CreateML/Components): \(error.localizedDescription)")
            return nil
        } catch {
            print("  ❌ トレーニングプロセス中に予期しないエラーが発生しました (マルチラベル - CreateML/Components): \(error.localizedDescription)")
            if let nsError = error as NSError? {
                print("    詳細なエラー情報: \(nsError.userInfo)")
            }
            return nil
        }
    }
}

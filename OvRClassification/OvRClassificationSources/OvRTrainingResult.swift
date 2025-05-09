import Foundation
import SCSInterface

public struct OvRTrainingResult: TrainingResultProtocol {
    public let modelName: String
    public let modelOutputPath: String
    public let reportPath: String
    public let oneLabelName: String
    public let restLabelNames: [String]
    public let positiveSamplesCount: Int
    public let negativeSamplesCount: Int
    public let trainingAccuracy: Double
    public let validationAccuracy: Double
    public let trainingError: Double
    public let validationError: Double
    public let trainingDuration: TimeInterval
    public let trainingDataPath: String

    public var classLabels: [String] {
        return ([oneLabelName] + restLabelNames).sorted()
    }

    public func saveLog(trainer: any ScreeningTrainerProtocol, modelAuthor: String, modelDescription: String, modelVersion: String) {
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd HH:mm:ss Z"
        let generatedDateString = dateFormatter.string(from: Date())

        let classLabelsString = self.classLabels.joined(separator: ", ")
        let restLabelsSummary = restLabelNames.isEmpty ? "該当なし" : restLabelNames.sorted().joined(separator: ", ")

        let trainingAccStr = String(format: "%.2f", trainingAccuracy * 100)
        let validationAccStr = String(format: "%.2f", validationAccuracy * 100)
        let trainingErrStr = String(format: "%.2f", trainingError * 100)
        let validationErrStr = String(format: "%.2f", validationError * 100)
        let durationStr = String(format: "%.2f", trainingDuration)

        let reportTitleOneLabelName = oneLabelName
        let specificModelDescription = "\(modelDescription) — '\(reportTitleOneLabelName)' とその他全てのクラスとの二値分類。"

        let reportContent = """
        # OvR 分類レポート: \(reportTitleOneLabelName) vs その他
        
        ## モデル詳細
        - **個別モデル名**: \(self.modelName) 
        - **保存先モデルパス**: \(self.modelOutputPath)
        - **レポート生成日時**: \(generatedDateString)
        
        ## トレーニング設定
        - **ターゲットラベル (One)**: \(reportTitleOneLabelName) (\(positiveSamplesCount) サンプル)
        - **その他のラベル (Rest)**: \(restLabelsSummary) (合計 \(negativeSamplesCount) サンプル)
        - **このOvRペアで考慮された全ラベル**: \(classLabelsString)
        
        ## パフォーマンス指標
        - **トレーニング所要時間**: \(durationStr) 秒
        - **トレーニング正解率**: \(trainingAccStr)%
        - **トレーニングエラー率**: \(trainingErrStr)%
        - **検証データ正解率**: \(validationAccStr)%
        - **検証データエラー率**: \(validationErrStr)%
        
        ## モデルメタデータ (.mlmodelに記述)
        - **作成者**: \(modelAuthor)
        - **説明**: \(specificModelDescription) 
        - **バージョン**: \(modelVersion)
        
        *このレポートは、'\(reportTitleOneLabelName)' を他の全てのラベルから区別するための二値分類器のトレーニングについて記述しています。*
        """
        do {
            try reportContent.write(to: URL(fileURLWithPath: self.reportPath), atomically: true, encoding: .utf8)
            print("  ✅ 個別OvRレポートを保存しました: \(self.reportPath)")
        } catch {
            print("  ❌ 個別OvRレポートの保存エラー (\(self.modelName)): \(error.localizedDescription)")
        }
    }
}

public struct OvRBatchResult {
    public let batchVersion: String
    public let individualResults: [OvRTrainingResult]
    public let mainOutputDirectoryPath: String

    public func saveLog(trainer: any ScreeningTrainerProtocol, modelAuthor: String, modelDescription: String, modelVersion: String) {
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd HH:mm:ss Z"
        let generatedDateString = dateFormatter.string(from: Date())

        var summaryContent = """
        # OvR バッチ分類レポート
        
        ## バッチ詳細
        - **バッチコーディネーターモデル名**: \(trainer.modelName) 
        - **バッチバージョン**: \(batchVersion)
        - **レポート生成日時**: \(generatedDateString)
        - **トレーニング済みOvRモデル総数**: \(individualResults.count)
        - **メイン出力ディレクトリ**: \(mainOutputDirectoryPath)

        ## モデルメタデータ (バッチ全体)
        - **作成者**: \(modelAuthor)
        - **説明**: \(modelDescription)
        - **バージョン**: \(modelVersion)
        
        ## 個別OvRモデル概要
        
        """

        if individualResults.isEmpty {
            summaryContent += "このバッチで正常にトレーニングされた個別のOvRモデルはありませんでした。\n"
        } else {
            for result in individualResults {
                let relativeModelPath = result.modelOutputPath.replacingOccurrences(of: mainOutputDirectoryPath, with: ".")
                let relativeReportPath = result.reportPath.replacingOccurrences(of: mainOutputDirectoryPath, with: ".")
                summaryContent += """
                ---
                - **ターゲットラベル (One)**: \(result.oneLabelName)
                  - **モデル名**: \(result.modelName)
                  - **モデルパス**: \(relativeModelPath)
                  - **レポートパス**: \(relativeReportPath)
                  - **トレーニング正解率**: \(result.trainingAccuracy != nil ? String(format: "%.2f", result.trainingAccuracy! * 100) : "該当なし")%
                  - **検証データ正解率**: \(result.validationAccuracy != nil ? String(format: "%.2f", result.validationAccuracy! * 100) : "該当なし")%
                
                """
            }
        }
        
        let summaryReportFileName = "OvR_Batch_Summary_\(batchVersion).md"
        let summaryReportURL = URL(fileURLWithPath: mainOutputDirectoryPath).appendingPathComponent(summaryReportFileName)

        do {
            try summaryContent.write(to: summaryReportURL, atomically: true, encoding: .utf8)
            print("✅ OvRバッチ概要レポートを保存しました: \(summaryReportURL.path)")
        } catch {
            print("❌ OvRバッチ概要レポートの保存エラー: \(error.localizedDescription)")
        }
    }
}

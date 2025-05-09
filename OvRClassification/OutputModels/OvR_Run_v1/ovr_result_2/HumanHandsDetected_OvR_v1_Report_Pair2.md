# OvR 分類レポート: HumanHandsDetected vs その他

## モデル詳細
- **個別モデル名**: HumanHandsDetected_OvR_v1.mlmodel 
- **保存先モデルパス**: /Users/akitora.hayashi/iOS Projects/ScaryCatScreeningML/OvRClassification/OutputModels/OvR_Run_v1/OvR_Result_2/HumanHandsDetected_OvR_v1.mlmodel
- **レポート生成日時**: 2025-05-09 12:51:51 +0900

## トレーニング設定
- **ターゲットラベル (One)**: HumanHandsDetected (60 サンプル)
- **その他のラベル (Rest)**: Rest (合計 91 サンプル)
- **このOvRペアで考慮された全ラベル**: HumanHandsDetected, Rest

## パフォーマンス指標
- **トレーニング所要時間**: 0.10 秒
- **トレーニング正解率**: 0.00%
- **トレーニングエラー率**: 0.00%
- **検証データ正解率**: 0.00%
- **検証データエラー率**: 0.00%

## モデルメタデータ (.mlmodelに記述)
- **作成者**: akitora
- **説明**: One-vs-Rest (OvR) Batch: ScaryCatScreener Training — 'HumanHandsDetected' とその他全てのクラスとの二値分類。 
- **バージョン**: v1

*このレポートは、'HumanHandsDetected' を他の全てのラベルから区別するための二値分類器のトレーニングについて記述しています。*
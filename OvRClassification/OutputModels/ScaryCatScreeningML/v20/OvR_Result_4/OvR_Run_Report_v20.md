# OvR (One-vs-Rest) トレーニング実行レポート

## 実行概要
モデル群         : OvRモデル群 (One-vs-Rest)
モデルベース名   : ScaryCatScreeningML
レポート生成日時   : 2025-05-21 19:12:20 +0900
最大反復回数     : 8 (各ペアモデル共通)
データ拡張       : なし
特徴抽出器       : ScenePrint(revision: 1)

## 個別 "One" モデルのパフォーマンス指標
| "One" クラス名 | モデル名 (vs Rest) | 検証正解率 | 再現率 | 適合率 |
|----------------|----------------------|--------------|----------|----------|
| Sphynx | ScaryCatScreeningML_OvR_Sphynx_vs_Rest_v20 | 8750.00% | 75.00% | 100.00% |
| HumanHandsDetected | ScaryCatScreeningML_OvR_HumanHandsDetected_vs_Rest_v20 | 6500.00% | 60.00% | 66.67% |
| BlackAndWhite | ScaryCatScreeningML_OvR_BlackAndWhite_vs_Rest_v20 | 9285.71% | 92.86% | 92.86% |
| MouthOpen | ScaryCatScreeningML_OvR_MouthOpen_vs_Rest_v20 | 5000.00% | 57.14% | 50.00% |

## 共通メタデータ
作成者            : akitora
バージョン        : v20
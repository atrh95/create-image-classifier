# OvR (One-vs-Rest) トレーニング実行レポート

## 実行概要
モデル群         : OvRモデル群 (One-vs-Rest)
モデルベース名   : ScaryCatScreeningML
レポート生成日時   : 2025-05-21 19:21:34 +0900
最大反復回数     : 15 (各ペアモデル共通)
データ拡張       : なし
特徴抽出器       : ScenePrint(revision: 2)

## 個別 "One" モデルのパフォーマンス指標
| "One" クラス名 | モデル名 (vs Rest) | 検証正解率 | 再現率 | 適合率 |
|----------------|----------------------|--------------|----------|----------|
| Sphynx | ScaryCatScreeningML_OvR_Sphynx_vs_Rest_v25 | 8750.00% | 100.00% | 80.00% |
| HumanHandsDetected | ScaryCatScreeningML_OvR_HumanHandsDetected_vs_Rest_v25 | 9000.00% | 90.00% | 90.00% |
| BlackAndWhite | ScaryCatScreeningML_OvR_BlackAndWhite_vs_Rest_v25 | 10000.00% | 100.00% | 100.00% |
| MouthOpen | ScaryCatScreeningML_OvR_MouthOpen_vs_Rest_v25 | 8571.43% | 85.71% | 85.71% |

## 共通メタデータ
作成者            : akitora
バージョン        : v25
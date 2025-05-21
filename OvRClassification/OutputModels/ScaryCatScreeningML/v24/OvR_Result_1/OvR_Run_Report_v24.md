# OvR (One-vs-Rest) トレーニング実行レポート

## 実行概要
モデル群         : OvRモデル群 (One-vs-Rest)
モデルベース名   : ScaryCatScreeningML
レポート生成日時   : 2025-05-21 19:19:57 +0900
最大反復回数     : 15 (各ペアモデル共通)
データ拡張       : なし
特徴抽出器       : ScenePrint(revision: 1)

## 個別 "One" モデルのパフォーマンス指標
| "One" クラス名 | モデル名 (vs Rest) | 検証正解率 | 再現率 | 適合率 |
|----------------|----------------------|--------------|----------|----------|
| Sphynx | ScaryCatScreeningML_OvR_Sphynx_vs_Rest_v24 | 8750.00% | 87.50% | 87.50% |
| HumanHandsDetected | ScaryCatScreeningML_OvR_HumanHandsDetected_vs_Rest_v24 | 8500.00% | 90.00% | 81.82% |
| BlackAndWhite | ScaryCatScreeningML_OvR_BlackAndWhite_vs_Rest_v24 | 9285.71% | 85.71% | 100.00% |
| MouthOpen | ScaryCatScreeningML_OvR_MouthOpen_vs_Rest_v24 | 7857.14% | 71.43% | 83.33% |

## 共通メタデータ
作成者            : akitora
バージョン        : v24
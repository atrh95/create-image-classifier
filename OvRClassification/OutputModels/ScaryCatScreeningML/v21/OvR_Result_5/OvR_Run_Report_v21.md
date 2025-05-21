# OvR (One-vs-Rest) トレーニング実行レポート

## 実行概要
モデル群         : OvRモデル群 (One-vs-Rest)
モデルベース名   : ScaryCatScreeningML
レポート生成日時   : 2025-05-21 19:14:13 +0900
最大反復回数     : 8 (各ペアモデル共通)
データ拡張       : なし
特徴抽出器       : ScenePrint(revision: 2)

## 個別 "One" モデルのパフォーマンス指標
| "One" クラス名 | モデル名 (vs Rest) | 検証正解率 | 再現率 | 適合率 |
|----------------|----------------------|--------------|----------|----------|
| Sphynx | ScaryCatScreeningML_OvR_Sphynx_vs_Rest_v21 | 9375.00% | 87.50% | 100.00% |
| HumanHandsDetected | ScaryCatScreeningML_OvR_HumanHandsDetected_vs_Rest_v21 | 8500.00% | 80.00% | 88.89% |
| BlackAndWhite | ScaryCatScreeningML_OvR_BlackAndWhite_vs_Rest_v21 | 9642.86% | 100.00% | 93.33% |
| MouthOpen | ScaryCatScreeningML_OvR_MouthOpen_vs_Rest_v21 | 7142.86% | 85.71% | 66.67% |

## 共通メタデータ
作成者            : akitora
バージョン        : v21
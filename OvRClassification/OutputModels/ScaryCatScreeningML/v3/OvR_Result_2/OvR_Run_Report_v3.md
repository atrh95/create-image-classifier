# OvR (One-vs-Rest) トレーニング実行レポート

## 実行概要
モデル群         : OvRモデル群 (One-vs-Rest)
モデルベース名   : ScaryCatScreeningML
レポート生成日時   : 2025-05-18 00:28:29 +0900
最大反復回数     : 10 (各ペアモデル共通)
データ拡張       : なし
特徴抽出器       : ScenePrint(revision: 1)

## 個別 "One" モデルのパフォーマンス指標
| "One" クラス名 | モデル名 (vs Rest) | 検証正解率 | 再現率 | 適合率 |
|----------------|----------------------|--------------|----------|----------|
| Sphynx | ScaryCatScreeningML_OvR_Sphynx_vs_Rest_vv3_idx0 | 9375.00% | 87.50% | 100.00% |
| HumanHandsDetected | ScaryCatScreeningML_OvR_HumanHandsDetected_vs_Rest_vv3_idx1 | 8000.00% | 85.71% | 75.00% |
| BlackAndWhite | ScaryCatScreeningML_OvR_BlackAndWhite_vs_Rest_vv3_idx2 | 9545.45% | 100.00% | 91.67% |
| MouthOpen | ScaryCatScreeningML_OvR_MouthOpen_vs_Rest_vv3_idx3 | 4166.67% | 50.00% | 42.86% |

## 共通メタデータ
作成者            : akitora
バージョン        : v3
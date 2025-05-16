# OvR (One-vs-Rest) トレーニング実行レポート

## 実行概要
モデル群         : OvRモデル群 (One-vs-Rest)
レポート生成日時   : 2025-05-16 18:54:39 +0900
最大反復回数     : 15 (各ペアモデル共通)
## 個別モデルのパフォーマンス指標
| モデル名 (PositiveClass) | 検証正解率 | 検証再現率 | 検証適合率 |
|--------------------------|--------------|--------------|--------------|
| OvR_Sphynx_vs_Rest (Sphynx) | 8750.00% | 75.00% | 100.00% |
| OvR_HumanHandsDetected_vs_Rest (HumanHandsDetected) | 7333.33% | 100.00% | 63.64% |
| OvR_BlackAndWhite_vs_Rest (BlackAndWhite) | 7727.27% | 72.73% | 80.00% |
| OvR_MouthOpen_vs_Rest (MouthOpen) | 6666.67% | 83.33% | 62.50% |

## 共通メタデータ
作成者            : akitora
全体説明          : ScaryCatScreener Training (このOvR実行全体に対して)
バージョン        : v3
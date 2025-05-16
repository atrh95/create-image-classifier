# OvR (One-vs-Rest) トレーニング実行レポート

## 実行概要
モデル群         : OvRモデル群 (One-vs-Rest)
レポート生成日時   : 2025-05-16 18:57:00 +0900
最大反復回数     : 15 (各ペアモデル共通)
## 個別モデルのパフォーマンス指標
| モデル名 (PositiveClass) | 検証正解率 | 検証再現率 | 検証適合率 |
|--------------------------|--------------|--------------|--------------|
| OvR_Sphynx_vs_Rest (Sphynx) | 93.75% | 87.50% | 100.00% |
| OvR_HumanHandsDetected_vs_Rest (HumanHandsDetected) | 86.67% | 85.71% | 85.71% |
| OvR_BlackAndWhite_vs_Rest (BlackAndWhite) | 81.82% | 72.73% | 88.89% |
| OvR_MouthOpen_vs_Rest (MouthOpen) | 66.67% | 66.67% | 66.67% |

## 共通メタデータ
作成者            : akitora
全体説明          : ScaryCatScreener Training (このOvR実行全体に対して)
バージョン        : v3
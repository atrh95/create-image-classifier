# OvR (One-vs-Rest) トレーニング実行レポート

## 実行概要
モデル群名         : OvR
レポート生成日時   : 2025-05-15 21:27:13 +0900
最大反復回数     : 15 (各ペアモデル共通)
## 個別モデルのパフォーマンス指標
| モデル名 (PositiveClass) | 検証正解率 | 検証再現率 | 検証適合率 | 説明 |
|--------------------------|--------------|--------------|--------------|------|
| OvR_Sphynx_vs_Rest (Sphynx) | 8750.00% | 0.00% | 0.00% | 訓練正解率: 100.0%, 検証正解率: 87.5%
陽性クラス: Sphynx, 再現率: 0.... |
| OvR_HumanHandsDetected_vs_Rest (HumanHandsDetected) | 8666.67% | 0.00% | 0.00% | 訓練正解率: 99.7%, 検証正解率: 86.7%
陽性クラス: HumanHandsDetect... |
| OvR_BlackAndWhite_vs_Rest (BlackAndWhite) | 8636.36% | 0.00% | 0.00% | 訓練正解率: 100.0%, 検証正解率: 86.4%
陽性クラス: BlackAndWhite, ... |
| OvR_MouthOpen_vs_Rest (MouthOpen) | 7500.00% | 0.00% | 0.00% | 訓練正解率: 100.0%, 検証正解率: 75.0%
陽性クラス: MouthOpen, 再現率:... |


### 個別モデル詳細説明:
- **OvR_Sphynx_vs_Rest (Sphynx)**: 訓練正解率: 100.0%, 検証正解率: 87.5%
陽性クラス: Sphynx, 再現率: 0.0%, 適合率: 0.0%
クラス構成: Sphynx: 76枚; Rest: 76枚
(検証: 自動分割)
- **OvR_HumanHandsDetected_vs_Rest (HumanHandsDetected)**: 訓練正解率: 99.7%, 検証正解率: 86.7%
陽性クラス: HumanHandsDetected, 再現率: 0.0%, 適合率: 0.0%
クラス構成: HumanHandsDetected: 150枚; Rest: 152枚
(検証: 自動分割)
- **OvR_BlackAndWhite_vs_Rest (BlackAndWhite)**: 訓練正解率: 100.0%, 検証正解率: 86.4%
陽性クラス: BlackAndWhite, 再現率: 0.0%, 適合率: 0.0%
クラス構成: BlackAndWhite: 222枚; Rest: 224枚
(検証: 自動分割)
- **OvR_MouthOpen_vs_Rest (MouthOpen)**: 訓練正解率: 100.0%, 検証正解率: 75.0%
陽性クラス: MouthOpen, 再現率: 0.0%, 適合率: 0.0%
クラス構成: MouthOpen: 113枚; Rest: 116枚
(検証: 自動分割)

## 共通メタデータ
作成者            : akitora
全体説明          : ScaryCatScreener Training (このOvR実行全体に対して)
バージョン        : v3
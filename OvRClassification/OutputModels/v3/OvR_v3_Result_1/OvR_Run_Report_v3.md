# OvR (One-vs-Rest) トレーニング実行レポート

## 実行概要
モデル群名         : OvR
レポート生成日時   : 2025-05-15 21:02:42 +0900
最大反復回数     : 15 (各ペアモデル共通)
## 個別モデルのパフォーマンス指標
| モデル名 (PositiveClass) | 検証正解率 | 検証再現率 | 検証適合率 | 説明 |
|--------------------------|--------------|--------------|--------------|------|
| OvR_Sphynx_vs_Rest_vv3.mlmodel (Sphynx) | 93.75% | 87.50% | 100.00% | 訓練正解率: 100.0%, 検証正解率: 93.8%, 再現率(Sphynx): 87.5%, 適... |
| OvR_HumanHandsDetected_vs_Rest_vv3.mlmodel (HumanHandsDetected) | 93.33% | 85.71% | 100.00% | 訓練正解率: 99.7%, 検証正解率: 93.3%, 再現率(HumanHandsDetected... |
| OvR_BlackAndWhite_vs_Rest_vv3.mlmodel (BlackAndWhite) | 81.82% | 81.82% | 81.82% | 訓練正解率: 100.0%, 検証正解率: 81.8%, 再現率(BlackAndWhite): 8... |
| OvR_MouthOpen_vs_Rest_vv3.mlmodel (MouthOpen) | 50.00% | 66.67% | 50.00% | 訓練正解率: 97.7%, 検証正解率: 50.0%, 再現率(MouthOpen): 66.7%,... |


### 個別モデル詳細説明:
- **OvR_Sphynx_vs_Rest_vv3.mlmodel (Sphynx)**: 訓練正解率: 100.0%, 検証正解率: 93.8%, 再現率(Sphynx): 87.5%, 適合率(Sphynx): 100.0%. サンプル (陽性/Rest): 76/76 (自動分割)
- **OvR_HumanHandsDetected_vs_Rest_vv3.mlmodel (HumanHandsDetected)**: 訓練正解率: 99.7%, 検証正解率: 93.3%, 再現率(HumanHandsDetected): 85.7%, 適合率(HumanHandsDetected): 100.0%. サンプル (陽性/Rest): 150/152 (自動分割)
- **OvR_BlackAndWhite_vs_Rest_vv3.mlmodel (BlackAndWhite)**: 訓練正解率: 100.0%, 検証正解率: 81.8%, 再現率(BlackAndWhite): 81.8%, 適合率(BlackAndWhite): 81.8%. サンプル (陽性/Rest): 222/224 (自動分割)
- **OvR_MouthOpen_vs_Rest_vv3.mlmodel (MouthOpen)**: 訓練正解率: 97.7%, 検証正解率: 50.0%, 再現率(MouthOpen): 66.7%, 適合率(MouthOpen): 50.0%. サンプル (陽性/Rest): 113/116 (自動分割)

## 共通メタデータ
作成者            : akitora
全体説明          : ScaryCatScreener Training (このOvR実行全体に対して)
バージョン        : v3
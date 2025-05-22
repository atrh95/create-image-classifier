# OvR (One-vs-Rest) トレーニング実行レポート

## 実行概要
モデル群         : OvRモデル群 (One-vs-Rest)
モデルベース名   : ScaryCatScreeningML
レポート生成日時   : 2025-05-22 12:47:08 +0900
最大反復回数     : 8 (各ペアモデル共通)
データ拡張       : なし
特徴抽出器       : ScenePrint(revision: 1)
検出されたクラス: sphynx, safe, human_hands_detected, black_and_white, mouth_open

## 個別ペアのトレーニング結果
## sphynx
- 訓練正解率: 100.0%
- 検証正解率: 93.8%
- 再現率 (Recall)    : 100.0%
- 適合率 (Precision) : 88.9%
- F1スコア          : 94.1%


Actual\Predicted	Positive	Negative
Positive	8	0
Negative	1	7

## human_hands_detected
- 訓練正解率: 91.6%
- 検証正解率: 90.0%
- 再現率 (Recall)    : 100.0%
- 適合率 (Precision) : 83.3%
- F1スコア          : 90.9%


Actual\Predicted	Positive	Negative
Positive	10	0
Negative	2	8

## black_and_white
- 訓練正解率: 97.0%
- 検証正解率: 92.9%
- 再現率 (Recall)    : 92.9%
- 適合率 (Precision) : 92.9%
- F1スコア          : 92.9%


Actual\Predicted	Positive	Negative
Positive	13	1
Negative	1	13

## mouth_open
- 訓練正解率: 94.5%
- 検証正解率: 71.4%
- 再現率 (Recall)    : 57.1%
- 適合率 (Precision) : 80.0%
- F1スコア          : 66.7%


Actual\Predicted	Positive	Negative
Positive	4	3
Negative	1	6


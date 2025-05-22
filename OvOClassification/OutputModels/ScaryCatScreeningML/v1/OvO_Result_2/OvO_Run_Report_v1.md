# OvO (One-vs-One) トレーニング実行レポート

## 実行概要
モデル群         : OvOモデル群 (One-vs-One)
モデルベース名   : ScaryCatScreeningML
レポート生成日時   : 2025-05-22 12:29:20 +0900
最大反復回数     : 8 (各ペアモデル共通)
データ拡張       : なし
特徴抽出器       : ScenePrint (revision: 1)
検出されたクラス: sphynx, human_hands_detected, black_and_white, mouth_open

## データソース
トレーニングデータディレクトリ: /Users/akitora.hayashi/iOS Projects/cat-screening-ml/TempOvOTrainingData/ScaryCatScreeningML_OvO_Sphynx_vs_HumanHandsDetected_v1_TempData_idx0; /Users/akitora.hayashi/iOS Projects/cat-screening-ml/TempOvOTrainingData/ScaryCatScreeningML_OvO_Sphynx_vs_BlackAndWhite_v1_TempData_idx1; /Users/akitora.hayashi/iOS Projects/cat-screening-ml/TempOvOTrainingData/ScaryCatScreeningML_OvO_Sphynx_vs_MouthOpen_v1_TempData_idx2; /Users/akitora.hayashi/iOS Projects/cat-screening-ml/TempOvOTrainingData/ScaryCatScreeningML_OvO_HumanHandsDetected_vs_BlackAndWhite_v1_TempData_idx3; /Users/akitora.hayashi/iOS Projects/cat-screening-ml/TempOvOTrainingData/ScaryCatScreeningML_OvO_HumanHandsDetected_vs_MouthOpen_v1_TempData_idx4; /Users/akitora.hayashi/iOS Projects/cat-screening-ml/TempOvOTrainingData/ScaryCatScreeningML_OvO_BlackAndWhite_vs_MouthOpen_v1_TempData_idx5
モデルファイル: /Users/akitora.hayashi/iOS Projects/cat-screening-ml/OvOClassification/OutputModels/ScaryCatScreeningML/v1/OvO_Result_2

# 個別ペアのトレーニング結果
### Sphynx_vs_HumanHandsDetected
- 訓練正解率: 98.8%
- 検証正解率: 100.0%
#### 混同行列分析
- 再現率 (Recall)    : 100.0%
- 適合率 (Precision) : 100.0%
- F1スコア          : 100.0%

#### 混同行列

Actual\Predicted	Positive	Negative
Positive	4	0
Negative	0	10

### Sphynx_vs_BlackAndWhite
- 訓練正解率: 98.0%
- 検証正解率: 94.4%
#### 混同行列分析
- 再現率 (Recall)    : 75.0%
- 適合率 (Precision) : 100.0%
- F1スコア          : 85.7%

#### 混同行列

Actual\Predicted	Positive	Negative
Positive	3	1
Negative	0	14

### Sphynx_vs_MouthOpen
- 訓練正解率: 100.0%
- 検証正解率: 90.9%
#### 混同行列分析
- 再現率 (Recall)    : 75.0%
- 適合率 (Precision) : 100.0%
- F1スコア          : 85.7%

#### 混同行列

Actual\Predicted	Positive	Negative
Positive	3	1
Negative	0	7

### HumanHandsDetected_vs_BlackAndWhite
- 訓練正解率: 93.3%
- 検証正解率: 87.5%
#### 混同行列分析
- 再現率 (Recall)    : 80.0%
- 適合率 (Precision) : 88.9%
- F1スコア          : 84.2%

#### 混同行列

Actual\Predicted	Positive	Negative
Positive	8	2
Negative	1	13

### HumanHandsDetected_vs_MouthOpen
- 訓練正解率: 93.7%
- 検証正解率: 76.5%
#### 混同行列分析
- 再現率 (Recall)    : 57.1%
- 適合率 (Precision) : 80.0%
- F1スコア          : 66.7%

#### 混同行列

Actual\Predicted	Positive	Negative
Positive	4	3
Negative	1	9

### BlackAndWhite_vs_MouthOpen
- 訓練正解率: 99.5%
- 検証正解率: 85.7%
#### 混同行列分析
- 再現率 (Recall)    : 85.7%
- 適合率 (Precision) : 75.0%
- F1スコア          : 80.0%

#### 混同行列

Actual\Predicted	Positive	Negative
Positive	6	1
Negative	2	12


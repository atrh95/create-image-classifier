# モデルトレーニング情報: ScaryCatScreeningML

## モデル詳細
モデル名           : ScaryCatScreeningML
ファイル生成日時   : 2025-05-22 14:21:08 +0900
最大反復回数     : 8 (注: CreateMLComponentsでは直接使用されません)
データ拡張       : なし
特徴抽出器       : ImageFeaturePrint

## トレーニング設定
アノテーションファイル: multilabel_cat_annotations.json
検出された全ラベル : black_and_white, human_hands_detected, mouth_open, sphynx

## 全体のパフォーマンス指標
トレーニング所要時間: 2.39 秒
トレーニング誤分類率 (学習時) : 100.00%
訓練データ正解率 (学習時) : 0.00%
検証データ正解率 (学習時自動検証) : 0.56%
検証誤分類率 (学習時自動検証) : 43.88%

## ラベル別性能指標
### black_and_white
再現率: 61.1%, 適合率: 91.7%, F1スコア: 73.3%
### human_hands_detected
再現率: 41.7%, 適合率: 100.0%, F1スコア: 58.8%
### mouth_open
再現率: 0.0%, 適合率: 0.0%, F1スコア: 0.0%
### sphynx
再現率: 92.3%, 適合率: 92.3%, F1スコア: 92.3%

## 混同行列
Label	True Positives	Total Actual
black_and_white	11	18
human_hands_detected	5	12
mouth_open	0	17
sphynx	12	13


## モデルメタデータ
作成者            : akitora
バージョン          : v1
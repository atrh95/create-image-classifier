# モデルトレーニング情報: ScaryCatScreeningML

## モデル詳細
モデル名           : ScaryCatScreeningML
ファイル生成日時   : 2025-05-22 13:39:08 +0900
最大反復回数     : 8
データ拡張       : なし
特徴抽出器       : ScenePrint(revision: 1)

## トレーニング設定
使用されたクラスラベル : black_and_white, human_hands_detected, mouth_open, sphynx

## パフォーマンス指標 (全体)
トレーニング所要時間: 3.25 秒
トレーニング誤分類率 (学習時) : 17.16%
訓練データ正解率 (学習時) : 0.83%
検証データ正解率 (学習時自動検証) : 0.50%
検証誤分類率 (学習時自動検証) : 50.00%## クラス別性能指標

### black_and_white
再現率: 60.0%, 適合率: 60.0%, F1スコア: 60.0%

### human_hands_detected
再現率: 33.3%, 適合率: 33.3%, F1スコア: 33.3%

### mouth_open
再現率: 25.0%, 適合率: 20.0%, F1スコア: 22.2%

### sphynx
再現率: 75.0%, 適合率: 100.0%, F1スコア: 85.7%

## 混同行列
Actual\Predicted	black_and_white	human_hands_detected	mouth_open	sphynx
black_and_white	3	0	2	0
human_hands_detected	0	1	2	0
mouth_open	1	2	1	0
sphynx	1	0	0	3

## モデルメタデータ
作成者            : akitora
バージョン          : v3
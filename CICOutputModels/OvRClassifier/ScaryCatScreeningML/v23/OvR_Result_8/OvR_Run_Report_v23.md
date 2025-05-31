# モデルトレーニング情報: ScaryCatScreeningML

## モデル詳細
モデル名           : ScaryCatScreeningML
ファイル生成日時   : 2025-05-31 13:03:21 +0900
最大反復回数     : 8
データ拡張       : なし
特徴抽出器       : ScenePrint

## トレーニング設定
使用されたクラスラベル : sphynx, safe, human_hands_detected, black_and_white, mouth_open

## パフォーマンス指標 (全体)
トレーニング所要時間: 1.76 秒
トレーニング誤分類率 (学習時) : 0.00%
訓練データ正解率 (学習時) : 1.00%
検証データ正解率 (学習時自動検証) : 0.94%
検証誤分類率 (学習時自動検証) : 6.25%## クラス別性能指標

### rest
再現率: 100.0%, 適合率: 88.9%, F1スコア: 94.1%

### sphynx
再現率: 87.5%, 適合率: 100.0%, F1スコア: 93.3%

## 混同行列
Actual\Predicted | rest | sphynx
rest | 8 | 0
sphynx | 1 | 7

## 個別モデルの性能指標


## モデルメタデータ
作成者            : akitora
バージョン          : v23
# モデルトレーニング情報: ScaryCatScreeningML

## モデル詳細
モデル名           : ScaryCatScreeningML
ファイル生成日時   : 2025-05-22 13:30:23 +0900
最大反復回数     : 8
データ拡張       : なし
特徴抽出器       : ScenePrint(revision: 1)

## トレーニング設定
使用されたクラスラベル : black_and_white, human_hands_detected, mouth_open, sphynx

## パフォーマンス指標 (全体)
トレーニング所要時間: 3.26 秒
トレーニング誤分類率 (学習時) : 18.15%
訓練データ正解率 (学習時) : 81.85%
検証データ正解率 (学習時自動検証) : 68.75%
検証誤分類率 (学習時自動検証) : 31.25%## クラス別性能指標
【black_and_white】
再現率: 60.0%, 適合率: 60.0%, F1スコア: 60.0%
【human_hands_detected】
再現率: 33.3%, 適合率: 50.0%, F1スコア: 40.0%
【mouth_open】
再現率: 75.0%, 適合率: 60.0%, F1スコア: 66.7%
【sphynx】
再現率: 100.0%, 適合率: 100.0%, F1スコア: 100.0%

## 混同行列
Actual\Predicted	black_and_white	human_hands_detected	mouth_open	sphynx
black_and_white	3	0	2	0
human_hands_detected	2	1	0	0
mouth_open	0	1	3	0
sphynx	0	0	0	4

## モデルメタデータ
作成者            : akitora
バージョン          : v3
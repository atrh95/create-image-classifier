# モデルトレーニング情報: ScaryCatScreeningML

## モデル詳細
モデル名           : ScaryCatScreeningML
ファイル生成日時   : 2025-06-09 01:02:43 +0900
最大反復回数     : 15
データ拡張       : ImageAugmentationOptions(rawValue: 18)
特徴抽出器       : ScenePrint - Logistic Regressor

## トレーニング設定
使用されたクラスラベル : black_and_white (297枚), safe (178枚), sphynx (185枚), human_hands_detected (263枚), mouth_open (219枚)
## 個別モデルの性能指標
| クラス | 訓練正解率 | 検証正解率 | 再現率 | 適合率 | F1スコア |
|--------|------------|------------|--------|--------|----------|
| sphynx | 98.9% | 94.1% | 100.0% | 88.9% | 0.941 |
| safe | 100.0% | 77.8% | 88.9% | 72.7% | 0.800 |
| human_hands_detected | 98.6% | 88.5% | 100.0% | 81.2% | 0.897 |
| black_and_white | 100.0% | 100.0% | 100.0% | 100.0% | 1.000 |
| mouth_open | 97.8% | 86.4% | 81.8% | 90.0% | 0.857 |

## モデルメタデータ
作成者            : akitora
バージョン          : v33
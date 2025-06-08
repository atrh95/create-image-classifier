# モデルトレーニング情報: ScaryCatScreeningML

## モデル詳細
モデル名           : ScaryCatScreeningML
ファイル生成日時   : 2025-06-09 01:04:10 +0900
最大反復回数     : 15
データ拡張       : ImageAugmentationOptions(rawValue: 18)
特徴抽出器       : ScenePrint - Logistic Regressor

## トレーニング設定
使用されたクラスラベル : black_and_white (297枚), sphynx (185枚), safe (178枚), human_hands_detected (263枚), mouth_open (219枚)
## 個別モデルの性能指標
| クラス | 訓練正解率 | 検証正解率 | 再現率 | 適合率 | F1スコア |
|--------|------------|------------|--------|--------|----------|
| sphynx | 98.9% | 94.4% | 100.0% | 90.0% | 0.947 |
| safe | 100.0% | 77.8% | 77.8% | 77.8% | 0.778 |
| human_hands_detected | 97.6% | 96.0% | 100.0% | 92.3% | 0.960 |
| black_and_white | 100.0% | 96.7% | 100.0% | 93.8% | 0.968 |
| mouth_open | 99.8% | 81.8% | 90.9% | 76.9% | 0.833 |

## モデルメタデータ
作成者            : akitora
バージョン          : v33
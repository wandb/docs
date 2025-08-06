---
title: 例 テーブル
description: W&B テーブル の例
menu:
  default:
    identifier: ja-guides-models-tables-tables-gallery
    parent: tables
---

以下のセクションでは、テーブルのいくつかの活用方法を紹介します。

### データの閲覧

モデルのトレーニングや評価時にメトリクスやリッチメディアをログとして記録し、その結果をクラウドや [ホスティングインスタンス]({{< relref path="/guides/hosting" lang="ja" >}}) に同期された永続的なデータベースで可視化できます。

{{< img src="/images/data_vis/tables_see_data.png" alt="データ閲覧テーブル" max-width="90%" >}}

たとえば、[写真データセットのバランスよい分割例](https://wandb.ai/stacey/mendeleev/artifacts/balanced_data/inat_80-10-10_5K/ab79f01e007113280018/files/data_split.table.json) をテーブルで確認できます。

### データをインタラクティブに探索

テーブルの閲覧、ソート、フィルタ、グループ化、結合、クエリで、データやモデルパフォーマンスを直感的に把握できます。静的ファイルを探したり分析スクリプトを再実行したりする必要はありません。

{{< img src="/images/data_vis/explore_data.png" alt="音声比較" max-width="90%">}}

たとえば、[スタイル変換音声](https://wandb.ai/stacey/cshanty/reports/Whale2Song-W-B-Tables-for-Audio--Vmlldzo4NDI3NzM) のレポートを参照できます。

### モデルバージョンの比較

異なるトレーニングエポック、データセット、ハイパーパラメータ設定、モデルアーキテクチャなど、さまざまな条件で結果を素早く比較できます。

{{< img src="/images/data_vis/compare_model_versions.png" alt="モデル比較" max-width="90%">}}

たとえば、[同じテスト画像に対する2つのモデルの比較テーブル](https://wandb.ai/stacey/evalserver_answers_2/artifacts/results/eval_Daenerys/c2290abd3d7274f00ad8/files/eval_results.table.json#b6dae62d4f00d31eeebf$eval_Bob) があります。

### 細部まで追跡し、全体像を把握

特定ステップでの予測をピンポイントで可視化できます。また、全体的な統計や誤りの傾向を特定して改善のヒントを探ることも可能です。1つのモデルのトレーニング中ステップ比較にも、異なるモデルバージョン間の比較にも、このツールが使えます。

{{< img src="/images/data_vis/track_details.png" alt="実験詳細のトラッキング" >}}

たとえば、[MNISTデータセットを1エポック後と5エポック後で分析するサンプルテーブル](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json#7dd0cd845c0edb469dec) があります。

## W&B Tables活用プロジェクト例
以下は、W&B Tablesを活用している実際のW&B Projectsの例です。

### 画像分類

[Visualize Data for Image Classification](https://wandb.ai/stacey/mendeleev/reports/Visualize-Data-for-Image-Classification--VmlldzozNjE3NjA) を読む、[data visualization nature Colab](https://wandb.me/dsviz-nature-colab) を試す、[artifacts context](https://wandb.ai/stacey/mendeleev/artifacts/val_epoch_preds/val_pred_gawf9z8j/2dcee8fa22863317472b/files/val_epoch_res.table.json) を探索することで、CNNが [iNaturalist](https://www.inaturalist.org/pages/developers) の写真から10種類の生物（植物、鳥、昆虫など）を識別する様子を確認できます。

{{< img src="/images/data_vis/image_classification.png" alt="2つの異なるモデルの予測で真のラベル分布を比較。" max-width="90%">}}

### 音声

[Whale2Song - W&B Tables for Audio](https://wandb.ai/stacey/cshanty/reports/Whale2Song-W-B-Tables-for-Audio--Vmlldzo4NDI3NzM) で音声テーブルとやり取りできます。録音したクジラの鳴き声と、バイオリンやトランペットなどの楽器で合成した同じメロディの音声を比較することが可能です。さらに、独自の曲を録音し、その合成バージョンをW&Bで [audio transfer Colab](http://wandb.me/audio-transfer) を使って探索できます。

{{< img src="/images/data_vis/audio.png" alt="音声テーブル例" max-width="90%">}}

### テキスト

トレーニングデータや生成出力のテキストサンプルを閲覧し、関連フィールドで動的にグループ化し、モデルバリアントや実験設定を横断して評価を揃えられます。テキストをMarkdownで表示したり、ビジュアル差分モードで比較したりできます。文字ベースRNNの例として [Shakespeare text generation report](https://wandb.ai/stacey/nlg/reports/Visualize-Text-Data-Predictions--Vmlldzo1NzcwNzY) を参照してください。

{{< img src="/images/data_vis/shakesamples.png" alt="隠れ層のサイズを倍にすると、より創造的なプロンプト補完が得られます。" max-width="90%">}}

### ビデオ

トレーニング中にログされたビデオを閲覧・集約して、モデルの理解に役立てます。これは、RLエージェントが [副作用の最小化](https://wandb.ai/stacey/saferlife/artifacts/video/videos_append-spawn/c1f92c6e27fa0725c154/files/video_examples.table.json) を目指す [SafeLifeベンチマーク](https://wandb.ai/safelife/v1dot2/benchmark) を使った初期事例です。

{{< img src="/images/data_vis/video.png" alt="成功したエージェントの例を簡単に閲覧" max-width="90%">}}

### 表形式データ

バージョン管理や重複排除を含めた[表形式データの分割と前処理方法のレポート](https://wandb.ai/dpaiton/splitting-tabular-data/reports/Tabular-Data-Versioning-and-Deduplication-with-Weights-Biases--VmlldzoxNDIzOTA1) を見ることができます。

{{< img src="/images/data_vis/tabs.png" alt="TablesとArtifactsによるワークフロー" max-width="90%">}}

### モデルバリアントの比較（セマンティックセグメンテーション）

セマンティックセグメンテーションでのTablesのログ例とモデル比較を掲載した [インタラクティブノートブック](https://wandb.me/dsviz-cars-demo) や [ライブ例](https://wandb.ai/stacey/evalserver_answers_2/artifacts/results/eval_Daenerys/c2290abd3d7274f00ad8/files/eval_results.table.json#a57f8e412329727038c2$eval_Ada) があります。[このTable](https://wandb.ai/stacey/evalserver_answers_2/artifacts/results/eval_Daenerys/c2290abd3d7274f00ad8/files/eval_results.table.json) でクエリを実際に試せます。

{{< img src="/images/data_vis/comparing_model_variants.png" alt="同じテストセットで2つのモデル間の最良予測を発見" max-width="90%" >}}

### トレーニング時間による改善の分析

[時間経過ごとの予測可視化レポート](https://wandb.ai/stacey/mnist-viz/reports/Visualize-Predictions-over-Time--Vmlldzo1OTQxMTk) と [インタラクティブノートブック](https://wandb.me/dsviz-mnist-colab) で詳細を解説しています。
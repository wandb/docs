---
title: 例 テーブル
description: W&B テーブルの例
menu:
  default:
    identifier: tables-gallery
    parent: tables
---

以下のセクションでは、Tables の活用例を紹介します。

### データを閲覧する

モデルのトレーニングや評価中にメトリクスやリッチメディアをログし、その結果をクラウドや[ホスティングインスタンス]({{< relref "/guides/hosting" >}})に同期された永続的なデータベースで可視化できます。

{{< img src="/images/data_vis/tables_see_data.png" alt="データ閲覧テーブル" max-width="90%" >}}

例えば、[写真データセットのバランス分割サンプル](https://wandb.ai/stacey/mendeleev/artifacts/balanced_data/inat_80-10-10_5K/ab79f01e007113280018/files/data_split.table.json)を示すこのテーブルをご覧ください。

### データをインタラクティブに探索する

テーブルを閲覧・並べ替え・フィルター・グループ化・結合・クエリすることで、データやモデルのパフォーマンスをより深く理解できます。静的ファイルを開いたり分析スクリプトを再実行する必要はありません。

{{< img src="/images/data_vis/explore_data.png" alt="音声比較" max-width="90%">}}

例えば、[スタイル転送済み音声](https://wandb.ai/stacey/cshanty/reports/Whale2Song-W-B-Tables-for-Audio--Vmlldzo4NDI3NzM)に関するこのレポートをご覧ください。

### モデルバージョンを比較する

異なるトレーニングエポック、データセット、ハイパーパラメータ、モデルアーキテクチャなどの結果を素早く比較できます。

{{< img src="/images/data_vis/compare_model_versions.png" alt="モデル比較" max-width="90%">}}

例えば、[同一テスト画像における2つのモデルの比較結果](https://wandb.ai/stacey/evalserver_answers_2/artifacts/results/eval_Daenerys/c2290abd3d7274f00ad8/files/eval_results.table.json#b6dae62d4f00d31eeebf$eval_Bob)を表示するこのテーブルをチェックしてみてください。

### 細部まで追跡し、全体像を把握する

特定のステップでの予測をズームインして可視化できますし、統計全体をズームアウトして誤りのパターンを特定したり、改善点を理解したりもできます。このツールは単一のモデルのトレーニングステップ比較にも、異なるモデルバージョンの結果比較にも役立ちます。

{{< img src="/images/data_vis/track_details.png" alt="実験詳細のトラッキング" >}}

例えば、[MNIST データセットにおいて 1 エポック後と 5 エポック後の比較分析](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json#7dd0cd845c0edb469dec)を行ったこの例をご覧ください。

## W&B Tables を活用したプロジェクト例

以下は、実際に W&B Tables を利用している W&B Projects の事例です。

### 画像分類

[Visualize Data for Image Classification](https://wandb.ai/stacey/mendeleev/reports/Visualize-Data-for-Image-Classification--VmlldzozNjE3NjA) を読んだり、[data visualization nature Colab](https://wandb.me/dsviz-nature-colab) を使ったり、[artifacts context](https://wandb.ai/stacey/mendeleev/artifacts/val_epoch_preds/val_pred_gawf9z8j/2dcee8fa22863317472b/files/val_epoch_res.table.json) を探ってみてください。CNN が [iNaturalist](https://www.inaturalist.org/pages/developers) の写真から（植物、鳥、昆虫など）10種類の生物をどのように分類しているかが分かります。

{{< img src="/images/data_vis/image_classification.png" alt="2モデル間の予測で真のラベル分布を比較" max-width="90%">}}

### オーディオ

[Whale2Song - W&B Tables for Audio](https://wandb.ai/stacey/cshanty/reports/Whale2Song-W-B-Tables-for-Audio--Vmlldzo4NDI3NzM) で、音色転送におけるオーディオテーブルを試せます。録音されたクジラの歌とバイオリンやトランペットのような楽器で再現されたメロディを比較できます。自分で歌を録音して [audio transfer Colab](http://wandb.me/audio-transfer) で合成バージョンを探ることも可能です。

{{< img src="/images/data_vis/audio.png" alt="オーディオテーブル例" max-width="90%">}}

### テキスト

トレーニングデータや生成出力のサンプルを閲覧し、関連フィールドで動的にグループ化したり、モデルバリアントや実験設定ごとに評価をそろえることができます。テキストは Markdown として表示したり、ビジュアル差分モードで比較したりできます。[Shakespeare テキスト生成レポート](https://wandb.ai/stacey/nlg/reports/Visualize-Text-Data-Predictions--Vmlldzo1NzcwNzY) では、文字単位 RNN の例を見ることができます。

{{< img src="/images/data_vis/shakesamples.png" alt="隠れ層を倍にすると よりクリエイティブなプロンプト補完が得られる" max-width="90%">}}

### ビデオ

トレーニング中にログしたビデオを閲覧・集約することで、モデルを理解できます。こちらは RL エージェントが[副作用最小化](https://wandb.ai/stacey/saferlife/artifacts/video/videos_append-spawn/c1f92c6e27fa0725c154/files/video_examples.table.json)を目指す [SafeLife ベンチマーク](https://wandb.ai/safelife/v1dot2/benchmark)での初期事例です。

{{< img src="/images/data_vis/video.png" alt="成功したエージェントのみを簡単に閲覧" max-width="90%">}}

### テーブルデータ

[テーブルデータの分割と前処理方法](https://wandb.ai/dpaiton/splitting-tabular-data/reports/Tabular-Data-Versioning-and-Deduplication-with-Weights-Biases--VmlldzoxNDIzOTA1)に関するレポートで、バージョン管理や重複除去を紹介しています。

{{< img src="/images/data_vis/tabs.png" alt="Tables と Artifacts のワークフロー" max-width="90%">}}

### モデルバリアントの比較（セマンティックセグメンテーション）

セマンティックセグメンテーションの Tables ロギングとモデル間比較の[インタラクティブノートブック](https://wandb.me/dsviz-cars-demo)や[ライブ例](https://wandb.ai/stacey/evalserver_answers_2/artifacts/results/eval_Daenerys/c2290abd3d7274f00ad8/files/eval_results.table.json#a57f8e412329727038c2$eval_Ada)があります。ぜひ [この Table](https://wandb.ai/stacey/evalserver_answers_2/artifacts/results/eval_Daenerys/c2290abd3d7274f00ad8/files/eval_results.table.json) で独自のクエリも試してみてください。

{{< img src="/images/data_vis/comparing_model_variants.png" alt="同一テストセットで2モデルの最良予測を見つける" max-width="90%" >}}

### トレーニング時間による改善の分析

[予測を時間軸で可視化](https://wandb.ai/stacey/mnist-viz/reports/Visualize-Predictions-over-Time--Vmlldzo1OTQxMTk)する詳細なレポートと、それに対応した[インタラクティブノートブック](https://wandb.me/dsviz-mnist-colab)もあります。
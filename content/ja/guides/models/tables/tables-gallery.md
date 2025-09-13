---
title: テーブルの例
description: W&B のテーブルの例
menu:
  default:
    identifier: ja-guides-models-tables-tables-gallery
    parent: tables
---

以下では、テーブルの活用方法の一部を紹介します。

### データを表示する

モデルのトレーニングや評価の最中にメトリクスやリッチメディアをログし、結果を クラウド と同期された永続的なデータベース、またはあなたの [ホスティングインスタンス]({{< relref path="/guides/hosting" lang="ja" >}}) で可視化できます。

{{< img src="/images/data_vis/tables_see_data.png" alt="データ閲覧用テーブル" max-width="90%" >}}

例えば、[写真データセットのバランス分割](https://wandb.ai/stacey/mendeleev/artifacts/balanced_data/inat_80-10-10_5K/ab79f01e007113280018/files/data_split.table.json) を示すこのテーブルを確認してください。

### データを対話的に探索する

テーブルを表示、ソート、フィルター、グループ化、結合、クエリして、データやモデルの性能を理解できます。静的ファイルを漁ったり、分析用スクリプトを再実行する必要はありません。

{{< img src="/images/data_vis/explore_data.png" alt="オーディオの比較" max-width="90%">}}

例えば、[スタイル転移したオーディオ](https://wandb.ai/stacey/cshanty/reports/Whale2Song-W-B-Tables-for-Audio--Vmlldzo4NDI3NzM) に関するこのレポートをご覧ください。

### モデルのバージョンを比較する

異なるトレーニングのエポック、データセット、ハイパーパラメーターの選択、モデルのアーキテクチャー などにまたがって、結果を素早く比較できます。

{{< img src="/images/data_vis/compare_model_versions.png" alt="モデル比較" max-width="90%">}}

例えば、[同じテスト画像上で 2 つのモデルを比較した例](https://wandb.ai/stacey/evalserver_answers_2/artifacts/results/eval_Daenerys/c2290abd3d7274f00ad8/files/eval_results.table.json#b6dae62d4f00d31eeebf$eval_Bob) のテーブルをご覧ください。

### 細部まで追跡しつつ、全体像も把握する

特定のステップにおける特定の予測をズームインして可視化できます。ズームアウトすれば、集計統計を確認し、エラーのパターンを特定し、改善の余地を把握できます。このツールは、単一のモデルのトレーニング内のステップ間比較にも、異なるモデルバージョン間の結果比較にも有効です。

{{< img src="/images/data_vis/track_details.png" alt="実験の詳細を追跡" >}}

例えば、[MNIST データセットで 1 エポック後と 5 エポック後の結果](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json#7dd0cd845c0edb469dec) を分析したサンプルテーブルをご覧ください。
## W&B Tables を使った W&B Projects の例
以下では、W&B Tables を活用している実際の W&B Projects をいくつか紹介します。

### 画像分類

[Visualize Data for Image Classification](https://wandb.ai/stacey/mendeleev/reports/Visualize-Data-for-Image-Classification--VmlldzozNjE3NjA) を読み、[データ可視化 Nature Colab](https://wandb.me/dsviz-nature-colab) を試すか、[Artifacts コンテキスト](https://wandb.ai/stacey/mendeleev/artifacts/val_epoch_preds/val_pred_gawf9z8j/2dcee8fa22863317472b/files/val_epoch_res.table.json) を探索して、CNN が [iNaturalist](https://www.inaturalist.org/pages/developers) の写真から植物・鳥・昆虫など 10 種類の生物をどのように識別するかを確認してください。

{{< img src="/images/data_vis/image_classification.png" alt="2 つの異なるモデルの予測間で真のラベル分布を比較" max-width="90%">}}

### オーディオ

音色変換のデモとして、[Whale2Song - W&B Tables for Audio](https://wandb.ai/stacey/cshanty/reports/Whale2Song-W-B-Tables-for-Audio--Vmlldzo4NDI3NzM) のオーディオテーブルと対話できます。記録したクジラの歌と、バイオリンやトランペットなどの楽器で同じメロディーを合成した音を比較できます。さらに、[audio transfer Colab](http://wandb.me/audio-transfer) を使って自分の歌を録音し、その合成版を W&B で探索することもできます。

{{< img src="/images/data_vis/audio.png" alt="オーディオテーブルの例" max-width="90%">}}

### テキスト

トレーニングデータや生成出力のテキストサンプルを閲覧し、関連フィールドで動的にグループ化し、モデルのバリアントや実験設定をまたいで評価を揃えられます。テキストを Markdown としてレンダリングしたり、ビジュアル差分モードで比較したりできます。文字ベースの RNN の例として、[Shakespeare のテキスト生成レポート](https://wandb.ai/stacey/nlg/reports/Visualize-Text-Data-Predictions--Vmlldzo1NzcwNzY) をご覧ください。

{{< img src="/images/data_vis/shakesamples.png" alt="隠れ層のサイズを 2 倍にすると、より創造的なプロンプト補完が得られる" max-width="90%">}}

### ビデオ

トレーニング中にログしたビデオを閲覧・集約して、モデルを理解しましょう。こちらは、[SafeLife ベンチマーク](https://wandb.ai/safelife/v1dot2/benchmark) を用いて、強化学習 (RL) のエージェントが[副作用を最小化](https://wandb.ai/stacey/saferlife/artifacts/video/videos_append-spawn/c1f92c6e27fa0725c154/files/video_examples.table.json) しようとする初期の例です。

{{< img src="/images/data_vis/video.png" alt="少数の成功したエージェントを簡単にブラウズ" max-width="90%">}}

### 表形式のデータ

バージョン管理と重複排除を使って[表形式データを分割・前処理する方法](https://wandb.ai/dpaiton/splitting-tabular-data/reports/Tabular-Data-Versioning-and-Deduplication-with-Weights-Biases--VmlldzoxNDIzOTA1) に関するレポートをご覧ください。

{{< img src="/images/data_vis/tabs.png" alt="テーブルと Artifacts のワークフロー" max-width="90%">}}

### モデルのバリアントの比較（セマンティックセグメンテーション）

セマンティックセグメンテーションのために Tables をログし、異なるモデルを比較するための [インタラクティブなノートブック](https://wandb.me/dsviz-cars-demo) と [ライブ例](https://wandb.ai/stacey/evalserver_answers_2/artifacts/results/eval_Daenerys/c2290abd3d7274f00ad8/files/eval_results.table.json#a57f8e412329727038c2$eval_Ada) です。このテーブルで自分のクエリを[試してみてください](https://wandb.ai/stacey/evalserver_answers_2/artifacts/results/eval_Daenerys/c2290abd3d7274f00ad8/files/eval_results.table.json)。

{{< img src="/images/data_vis/comparing_model_variants.png" alt="同じテストセットで 2 つのモデル間の最良の予測を見つける" max-width="90%" >}}

### トレーニング時間に伴う改善の分析

[時間経過に伴う予測を可視化](https://wandb.ai/stacey/mnist-viz/reports/Visualize-Predictions-over-Time--Vmlldzo1OTQxMTk) する方法に関する詳細なレポートと、対応する [インタラクティブなノートブック](https://wandb.me/dsviz-mnist-colab)。
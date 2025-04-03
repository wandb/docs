---
title: Example tables
description: W&B テーブル の例
menu:
  default:
    identifier: ja-guides-models-tables-tables-gallery
    parent: tables
---

次のセクションでは、テーブルを使用できる方法のいくつかを紹介します。

### データの表示

モデルトレーニングまたは評価中に、メトリクスとリッチメディアをログに記録し、クラウドまたは [ホスティングインスタンス]({{< relref path="/guides/hosting" lang="ja" >}}) に同期された永続的なデータベースで結果を可視化します。

{{< img src="/images/data_vis/tables_see_data.png" alt="データの例を参照し、カウントと分布を確認します" max-width="90%" >}}

たとえば、[写真データセットのバランスの取れた分割を示すこのテーブル](https://wandb.ai/stacey/mendeleev/artifacts/balanced_data/inat_80-10-10_5K/ab79f01e007113280018/files/data_split.table.json) を確認してください。

### データをインタラクティブに探索する

テーブルの表示、並べ替え、フィルタリング、グループ化、結合、およびクエリを実行して、データとモデルのパフォーマンスを理解します。静的なファイルを参照したり、分析スクリプトを再実行したりする必要はありません。

{{< img src="/images/data_vis/explore_data.png" alt="オリジナルの曲とシンセサイズされたバージョン（音色転送あり）を聴きます" max-width="90%">}}

たとえば、[スタイルが転送されたオーディオに関するこのレポート](https://wandb.ai/stacey/cshanty/reports/Whale2Song-W-B-Tables-for-Audio--Vmlldzo4NDI3NzM) を参照してください。

### モデルのバージョンを比較する

さまざまなトレーニングエポック、データセット、ハイパーパラメーターの選択、モデルアーキテクチャーなどで、結果をすばやく比較します。

{{< img src="/images/data_vis/compare_model_versions.png" alt="詳細な違いを確認する：左のモデルは赤い歩道を検出し、右のモデルは検出しない" max-width="90%">}}

たとえば、[同じテスト画像で 2 つのモデルを比較するこのテーブル](https://wandb.ai/stacey/evalserver_answers_2/artifacts/results/eval_Daenerys/c2290abd3d7274f00ad8/files/eval_results.table.json#b6dae62d4f00d31eeebf$eval_Bob) を参照してください。

### すべての詳細を追跡し、全体像を把握する

特定のステップで特定の予測を可視化するためにズームインします。集計統計を表示し、エラーのパターンを特定し、改善の機会を理解するためにズームアウトします。このツールは、単一のモデルトレーニングのステップを比較したり、異なるモデルバージョン間で結果を比較したりするために使用できます。

{{< img src="/images/data_vis/track_details.png" alt="" >}}

たとえば、[MNIST データセットで 1 回、次に 5 回のエポック後の結果を分析するこのサンプルテーブル](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json#7dd0cd845c0edb469dec) を参照してください。

## W&B Tables を使用したプロジェクト例

以下は、W&B Tables を使用する実際の W&B のProjectsのハイライトです。

### 画像分類

[このレポート](https://wandb.ai/stacey/mendeleev/reports/Visualize-Data-for-Image-Classification--VmlldzozNjE3NjA) を読むか、[この colab](https://wandb.me/dsviz-nature-colab) に従うか、[artifacts コンテキスト](https://wandb.ai/stacey/mendeleev/artifacts/val_epoch_preds/val_pred_gawf9z8j/2dcee8fa22863317472b/files/val_epoch_res.table.json) を調べて、CNN が [iNaturalist](https://www.inaturalist.org/pages/developers) の写真から 10 種類の生物（植物、鳥、昆虫など）を識別する方法を確認してください。

{{< img src="/images/data_vis/image_classification.png" alt="2 つの異なるモデルの予測にわたる真のラベルの分布を比較します。" max-width="90%">}}

### オーディオ

[音色転送に関するこのレポート](https://wandb.ai/stacey/cshanty/reports/Whale2Song-W-B-Tables-for-Audio--Vmlldzo4NDI3NzM) でオーディオテーブルを操作します。録音されたクジラの歌と、バイオリンやトランペットなどの楽器で同じメロディーをシンセサイズした演奏を比較できます。[この colab](http://wandb.me/audio-transfer) を使用して、独自の曲を録音し、W&B でシンセサイズされたバージョンを探索することもできます。

{{< img src="/images/data_vis/audio.png" alt="" max-width="90%">}}

### テキスト

トレーニングデータまたは生成された出力からテキストサンプルを参照し、関連フィールドで動的にグループ化し、モデルのバリアントまたは実験設定全体で評価を調整します。テキストを Markdown としてレンダリングするか、ビジュアル差分モードを使用してテキストを比較します。[このレポート](https://wandb.ai/stacey/nlg/reports/Visualize-Text-Data-Predictions--Vmlldzo1NzcwNzY) で、シェイクスピアを生成するための単純な文字ベースの RNN を探索してください。

{{< img src="/images/data_vis/shakesamples.png" alt="隠れ層のサイズを 2 倍にすると、より創造的なプロンプト補完が得られます。" max-width="90%">}}

### ビデオ

トレーニング中にログに記録されたビデオを参照および集計して、モデルを理解します。[副作用を最小限に抑える](https://wandb.ai/stacey/saferlife/artifacts/video/videos_append-spawn/c1f92c6e27fa0725c154/files/video_examples.table.json) ことを目指す RL エージェント向けの [SafeLife ベンチマーク](https://wandb.ai/safelife/v1dot2/benchmark) を使用した初期の例を次に示します。

{{< img src="/images/data_vis/video.png" alt="成功したエージェントを簡単に参照できます" max-width="90%">}}

### 表形式データ

バージョン管理と重複排除を使用して [表形式データを分割および事前処理する方法に関するレポート](https://wandb.ai/dpaiton/splitting-tabular-data/reports/Tabular-Data-Versioning-and-Deduplication-with-Weights-Biases--VmlldzoxNDIzOTA1) を表示します。

{{< img src="/images/data_vis/tabs.png" alt="Tables と Artifacts は連携して、データセットのイテレーションをバージョン管理、ラベル付け、および重複排除します" max-width="90%">}}

### モデルバリアントの比較（セマンティックセグメンテーション）

セマンティックセグメンテーション用の Tables のログを記録し、異なるモデルを比較する [インタラクティブノートブック](https://wandb.me/dsviz-cars-demo) と [ライブ例](https://wandb.ai/stacey/evalserver_answers_2/artifacts/results/eval_Daenerys/c2290abd3d7274f00ad8/files/eval_results.table.json#a57f8e412329727038c2$eval_Ada)。[この Table](https://wandb.ai/stacey/evalserver_answers_2/artifacts/results/eval_Daenerys/c2290abd3d7274f00ad8/files/eval_results.table.json) で独自のクエリを試してください。

{{< img src="/images/data_vis/comparing_model_variants.png" alt="同じテストセットで 2 つのモデル間で最適な予測を見つける" max-width="90%" >}}

### トレーニング時間に対する改善の分析

[時間経過に伴う予測の可視化](https://wandb.ai/stacey/mnist-viz/reports/Visualize-Predictions-over-Time--Vmlldzo1OTQxMTk) と、それに付随する [インタラクティブノートブック](https://wandb.me/dsviz-mnist-colab) に関する詳細なレポート。

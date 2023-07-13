---
slug: /guides/data-vis
description: Iterate on datasets and understand model predictions
displayed_sidebar: ja
---

# データ可視化

**W＆B Tables**を使用して、表形式のデータを記録、クエリ、および分析します。データセットを理解し、モデルの予測を可視化し、中央ダッシュボードで洞察を共有します。

* モデル、エポック、または個々の例にわたる変更を正確に比較する
* データの高次元のパターンを理解する
* 視覚サンプルで洞察を記録し、共有します


## W&B Tablesとは何ですか？

W＆Bテーブル（`wandb.Table`）は、各列にデータの単一のタイプが含まれる2次元のデータグリッドです。これは、より強力なDataFrameと考えてください。テーブルは、プリミティブおよび数値タイプ、およびネストされたリスト、辞書、リッチメディアタイプをサポートしています。 W＆Bにテーブルを記録し、UIでクエリ、比較、および分析します。

テーブルは、データセットからモデルの予測まで、MLワークフローに不可欠なあらゆる形式のデータを保存、理解、共有するのに適しています。

## なぜテーブルを使うのか？

### データを表示する

モデルのトレーニングや評価中にメトリクスやリッチメディアをログし、結果をクラウドに同期した永続的なデータベース、または[ホスティングインスタンス](https://docs.wandb.ai/guides/hosting)に可視化します。たとえば、この[写真データセットのバランスの取れた分割→](https://wandb.ai/stacey/mendeleev/artifacts/balanced\_data/inat\_80-10-10\_5K/ab79f01e007113280018/files/data\_split.table.json)を確認してください。

![実際の例を参照し、データのカウントと分布を確認する](/images/data_vis/tables_see_data.png)

### インタラクティブにデータを探索する

データとモデルのパフォーマンスを理解するために、テーブルを表示、ソート、フィルタリング、グループ化、結合、クエリーすることができます。静的なファイルを閲覧したり、分析スクリプトを再実行したりする必要はありません。例えば、このプロジェクトでは[スタイル変換オーディオ →](https://wandb.ai/stacey/cshanty/reports/Whale2Song-W-B-Tables-for-Audio--Vmlldzo4NDI3NzM) をご覧ください。 

![オリジナルの曲とそれらの合成バージョン（音色変換付き）を聴く](/images/data_vis/explore_data.png)

### モデルのバージョンを比較する

さまざまなトレーニングエポック、データセット、ハイパーパラメーターの選択、モデルアーキテクチャーなどの間で、すばやく結果を比較できます。例えば、同じテスト画像の[2つのモデルの比較 →](https://wandb.ai/stacey/evalserver_answers_2/artifacts/results/eval_Daenerys/c2290abd3d7274f00ad8/files/eval_results.table.json#b6dae62d4f00d31eeebf$eval_Bob)を見てみましょう。

![詳細な違いを見る：左のモデルは赤い歩道を検出しますが、右のモデルは検出しません。](/images/data_vis/compare_model_versions.png)

### 詳細を追跡し、全体像を把握する

特定のステップでの特定の予測を視覚化するためにズームインします。ズームアウトして、集計された統計を表示し、エラーのパターンを特定し、改善の機会を理解します。このツールは、単一のモデルトレーニングのステップを比較する場合や、異なるモデルバージョン間の結果を比較する場合にも使用できます。例として、MNISTで1エポックと5エポックの後の結果を分析したテーブルをご覧ください[→](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json#7dd0cd845c0edb469dec)

![](/images/data_vis/track_details.png)

## W&Bテーブルを使用した例プロジェクト

### 画像分類

このレポートを[読む](https://wandb.ai/stacey/mendeleev/reports/Visualize-Data-for-Image-Classification--VmlldzozNjE3NjA)、このColabを[たどる](https://wandb.me/dsviz-nature-colab)、またはこの[アーティファクトコンテキスト](https://wandb.ai/stacey/mendeleev/artifacts/val_epoch_preds/val_pred_gawf9z8j/2dcee8fa22863317472b/files/val_epoch_res.table.json)を調べて、[iNaturalist](https://www.inaturalist.org/pages/developers)の写真から10種類の生き物（植物、鳥、昆虫など）を識別するCNNを見てください。

![2つの異なるモデルの予測で、真のラベルの分布を比較します。](/images/data_vis/image_classification.png)

### オーディオ

タンバートランスファーに関する[このレポート](https://wandb.ai/stacey/cshanty/reports/Whale2Song-W-B-Tables-for-Audio--Vmlldzo4NDI3NzM)でオーディオテーブルを操作してみてください。録音されたクジラの歌と、バイオリンやトランペットのような楽器で合成された同じメロディーを比較することができます。また、自分の歌を録音して、W&Bで[このColab→](http://wandb.me/audio-transfer)を使って合成バージョンを探索することもできます。

![](/images/data_vis/audio.png)

### テキスト

トレーニングデータや生成された出力のテキストサンプルを参照し、関連するフィールドで動的にグループ化し、モデルのバリアントや実験設定を比較して評価を行うことができます。テキストをMarkdownとしてレンダリングするか、テキストを比較するためのビジュアル差分モードを使用します。[このレポート→](https://wandb.ai/stacey/nlg/reports/Visualize-Text-Data-Predictions--Vmlldzo1NzcwNzY)でシェイクスピアを生成するためのシンプルな文字ベースのRNNを探索してください。

![隠れ層のサイズを2倍にすると、より創造的なプロンプトの補完が得られます。](@site/static/images/data_vis/shakesamples.png)

### ビデオ

トレーニング中に記録されたビデオを閲覧し、集約してモデルを理解します。[SafeLifeベンチマーク](https://wandb.ai/safelife/v1dot2/benchmark) を使用した初期の例で、RLエージェントは[副作用を最小限に抑える](https://wandb.ai/stacey/saferlife/artifacts/video/videos_append-spawn/c1f92c6e27fa0725c154/files/video_examples.table.json)ことを目指しています。

![成功したエージェントを簡単に閲覧](/images/data_vis/video.png)

### 表形式のデータ

バージョン管理と重複排除を用いた[表形式データの分割と前処理](https://wandb.ai/dpaiton/splitting-tabular-data/reports/Tabular-Data-Versioning-and-Deduplication-with-Weights-Biases--VmlldzoxNDIzOTA1)に関するレポート。

![Tables and Artifactsは、バージョン管理、ラベル付け、およびデータセットの反復処理を重複排除するために協力します](@site/static/images/data_vis/tabs.png)

### モデルバリアントの比較（セマンティックセグメンテーション）

インタラクティブな[ノートブック](https://wandb.me/dsviz-cars-demo)と、セマンティックセグメンテーションのTablesをログして異なるモデルを比較する[ライブ例](https://wandb.ai/stacey/evalserver_answers_2/artifacts/results/eval_Daenerys/c2290abd3d7274f00ad8/files/eval_results.table.json#a57f8e412329727038c2$eval_Ada)。自分でクエリを試すには、[このテーブルで→](https://wandb.ai/stacey/evalserver_answers_2/artifacts/results/eval_Daenerys/c2290abd3d7274f00ad8/files/eval_results.table.json)

![同じテストセットに対する2つのモデルの最高の予測を見つける](/images/data_vis/comparing_model_variants.png)

### トレーニング時間をかけての改善の分析

[時間経過に伴う予測の可視化](https://wandb.ai/stacey/mnist-viz/reports/Visualize-Predictions-over-Time--Vmlldzo1OTQxMTk)に関する詳細なレポートと、それに伴う[インタラクティブなノートブック →](https://wandb.me/dsviz-mnist-colab)
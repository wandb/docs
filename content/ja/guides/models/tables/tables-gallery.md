---
title: 例 テーブル
description: W&B テーブル の例
menu:
  default:
    identifier: ja-guides-models-tables-tables-gallery
    parent: tables
---

以下のセクションでは、テーブルの使用方法の一部を紹介します。

### データを表示する

モデルのトレーニングまたは評価中にメトリクスやリッチメディアをログに記録し、それをクラウドに同期された永続的なデータベース、または[ホスティングインスタンス]({{< relref path="/guides/hosting" lang="ja" >}})で結果を視覚化します。

{{< img src="/images/data_vis/tables_see_data.png" alt="例を閲覧し、データのカウントと分布を検証する" max-width="90%" >}}

たとえば、このテーブルをご覧ください。[写真データセットのバランスの取れた分割](https://wandb.ai/stacey/mendeleev/artifacts/balanced_data/inat_80-10-10_5K/ab79f01e007113280018/files/data_split.table.json)を示しています。

### データを対話的に探索する

テーブルを表示、ソート、フィルタ、グループ化、結合、クエリして、データとモデルのパフォーマンスを理解します。静的ファイルを閲覧したり、分析スクリプトを再実行する必要はありません。

{{< img src="/images/data_vis/explore_data.png" alt="オリジナルの曲とその合成バージョン（ティンバー転送）を聴く" max-width="90%" >}}

たとえば、このレポートをご覧ください。[スタイルが転送されたオーディオ](https://wandb.ai/stacey/cshanty/reports/Whale2Song-W-B-Tables-for-Audio--Vmlldzo4NDI3NzM)について。

### モデルバージョンを比較する

異なるトレーニングエポック、データセット、ハイパーパラメーターの選択、モデルアーキテクチャーなど、さまざまな結果を迅速に比較します。

{{< img src="/images/data_vis/compare_model_versions.png" alt="細かな違いを確認: 左のモデルは赤い歩道を検出し、右のモデルは検出しない。" max-width="90%" >}}

たとえば、このテーブルを確認してください。[同じテスト画像で2つのモデルを比較](https://wandb.ai/stacey/evalserver_answers_2/artifacts/results/eval_Daenerys/c2290abd3d7274f00ad8/files/eval_results.table.json#b6dae62d4f00d31eeebf$eval_Bob)しています。

### すべての詳細を追跡し、大局を把握する

特定のステップでの特定の予測を視覚化するためにズームインします。ズームアウトして集計統計を確認し、エラーのパターンを識別し、改善の機会を理解します。このツールは、単一のモデルトレーニングからのステップを比較したり、異なるモデルバージョンの結果を比較するために使用されます。

{{< img src="/images/data_vis/track_details.png" alt="" >}}

たとえば、[1回とその後5回のエポック後のMNISTデータセットでの結果を分析](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json#7dd0cd845c0edb469dec)する例のテーブルをご覧ください。

## W&B Tablesを使用したプロジェクトの例
以下では、W&B Tablesを使用した実際のW&Bプロジェクトの例を紹介します。

### 画像分類

[このレポート](https://wandb.ai/stacey/mendeleev/reports/Visualize-Data-for-Image-Classification--VmlldzozNjE3NjA)を読み、[このcolab](https://wandb.me/dsviz-nature-colab)に従うか、この[アーティファクトコンテキスト](https://wandb.ai/stacey/mendeleev/artifacts/val_epoch_preds/val_pred_gawf9z8j/2dcee8fa22863317472b/files/val_epoch_res.table.json)を探り、CNNが[iNaturalist](https://www.inaturalist.org/pages/developers)の写真から10種類の生物（植物、鳥、昆虫など）を識別する方法を見てみてください。

{{< img src="/images/data_vis/image_classification.png" alt="2つの異なるモデルの予測に対する真のラベルの分布を比較する。" max-width="90%" >}}

### オーディオ

ティンバー転送に関する[このレポート](https://wandb.ai/stacey/cshanty/reports/Whale2Song-W-B-Tables-for-Audio--Vmlldzo4NDI3NzM)でオーディオテーブルと対話します。録音されたクジラの歌と同じメロディをバイオリンやトランペットのような楽器で合成したバージョンを比較できます。この[colab](http://wandb.me/audio-transfer)で自分の曲を録音し、その合成バージョンをW&Bで探索することもできます。

{{< img src="/images/data_vis/audio.png" alt="" max-width="90%">}}

### テキスト

トレーニングデータや生成された出力からテキストサンプルを閲覧し、関連するフィールドで動的にグループ化し、モデルバリエーションや実験設定に合わせて評価を整えます。Markdownとしてテキストをレンダリングするか、ビジュアル差分モードを使用してテキストを比較します。[このレポート](https://wandb.ai/stacey/nlg/reports/Visualize-Text-Data-Predictions--Vmlldzo1NzcwNzY)でシェイクスピアを生成するためのシンプルな文字ベースのRNNを探ります。

{{< img src="/images/data_vis/shakesamples.png" alt="隠れ層のサイズを倍増させると、より創造的なプロンプトの完了が得られる。" max-width="90%">}}

### ビデオ

モデルを理解するためにトレーニング中にログに記録されたビデオを閲覧および集約します。ここに、RLエージェントが[副作用を最小化](https://wandb.ai/stacey/saferlife/artifacts/video/videos_append-spawn/c1f92c6e27fa0725c154/files/video_examples.table.json)しようとするための[SafeLife ベンチマーク](https://wandb.ai/safelife/v1dot2/benchmark)を使用した初期の例があります。

{{< img src="/images/data_vis/video.png" alt="数少ない成功したエージェントを簡単に閲覧する" max-width="90%">}}

### 表形式データ

バージョン管理とデータの重複排除を使用して[表形式データを分割および前処理する](https://wandb.ai/dpaiton/splitting-tabular-data/reports/Tabular-Data-Versioning-and-Deduplication-with-Weights-Biases--VmlldzoxNDIzOTA1)方法に関するレポートを参照してください。

{{< img src="/images/data_vis/tabs.png" alt="Tables and Artifactsが連携してバージョンコントロール、ラベル付け、データセットの重複排除を行う" max-width="90%">}}

### モデルバリエーションの比較（セマンティックセグメンテーション）

セマンティックセグメンテーションのためのTablesをログに記録し、さまざまなモデルを比較する[対話型ノートブック](https://wandb.me/dsviz-cars-demo)と[ライブ例](https://wandb.ai/stacey/evalserver_answers_2/artifacts/results/eval_Daenerys/c2290abd3d7274f00ad8/files/eval_results.table.json)です。この[Table](https://wandb.ai/stacey/evalserver_answers_2/artifacts/results/eval_Daenerys/c2290abd3d7274f00ad8/files/eval_results.table.json)で自分のクエリを試してみてください。

{{< img src="/images/data_vis/comparing_model_variants.png" alt="同じテストセットに対する2つのモデル間で最高の予測を見つける" max-width="90%" >}}

### トレーニング時間の改善を分析する

[予測を時間にわたって視覚化する方法についての詳細なレポート](https://wandb.ai/stacey/mnist-viz/reports/Visualize-Predictions-over-Time--Vmlldzo1OTQxMTk)と、付随する[対話型ノートブック](https://wandb.me/dsviz-mnist-colab)です。
---
title: Example tables
description: W&B テーブル の例
menu:
  default:
    identifier: ja-guides-core-tables-tables-gallery
    parent: tables
---

以下のセクションは、Tables の使用方法のいくつかを強調しています。

### データを表示する

モデル トレーニングまたは評価中に メトリクス やリッチメディアをログに記録し、クラウドまたは [ホスティングインスタンス]({{< relref path="/guides/hosting" lang="ja" >}}) に同期された永続的なデータベースで結果を視覚化します。 

{{< img src="/images/data_vis/tables_see_data.png" alt="データの例を閲覧し、データの数と分布を確認する" max-width="90%" >}}

たとえば、[写真データセットのバランスの取れた分割](https://wandb.ai/stacey/mendeleev/artifacts/balanced_data/inat_80-10-10_5K/ab79f01e007113280018/files/data_split.table.json) を示すこのテーブルをチェックしてください。

### データを対話的に探索する

データとモデルのパフォーマンスを把握するために、テーブルを確認、並べ替え、フィルタリング、グループ化、結合、クエリを実行します。静的ファイルを参照したり、分析スクリプトを再実行したりする必要はありません。

 {{< img src="/images/data_vis/explore_data.png" alt="オリジナルの曲と合成版の曲を（ティンバートランスファーで）聴く" max-width="90%">}}

たとえば、[スタイル転送されたオーディオ](https://wandb.ai/stacey/cshanty/reports/Whale2Song-W-B-Tables-for-Audio--Vmlldzo4NDI3NzM) に関するこの Report を参照してください。

### モデルバージョンを比較する 

異なるトレーニングエポック、データセット、ハイパーパラメータの選択、モデルアーキテクチャーなどでの結果をすばやく比較します。

{{< img src="/images/data_vis/compare_model_versions.png" alt="詳細な違いを見る: 左のモデルはいくつかの赤い歩道を検出しますが、右のモデルは検出しません。" max-width="90%">}}

たとえば、[同じテスト画像で 2 つのモデルを比較する](https://wandb.ai/stacey/evalserver_answers_2/artifacts/results/eval_Daenerys/c2290abd3d7274f00ad8/files/eval_results.table.json#b6dae62d4f00d31eeebf$eval_Bob) このテーブルを参照してください。

### あらゆる詳細を追跡し、大きな絵を描く

特定のステップで特定の予測を視覚化するためにズームインします。ズームアウトして、集計統計を確認し、エラーのパターンを特定し、改善の機会を理解します。このツールは、単一のモデル トレーニングからのステップを比較する場合や、異なるモデルバージョン間での結果を比較する場合に機能します。

{{< img src="/images/data_vis/track_details.png" alt="" >}}

たとえば、[MNISTデータセットで1回、その後5回のエポック後](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json#7dd0cd845c0edb469dec) の結果を分析するこの例のテーブルを参照してください。

## W&B Tables を使用したプロジェクトの例
以下は、W&B Tables を使用した実際の W&B Projects のいくつかを紹介しています。

### 画像分類

[CNN が [iNaturalist](https://www.inaturalist.org/pages/developers) の写真から 10 種類の生物（植物、鳥、昆虫など）を識別する方法を確認するには、[このレポート](https://wandb.ai/stacey/mendeleev/reports/Visualize-Data-for-Image-Classification--VmlldzozNjE3NjA) を読み、[この colab](https://wandb.me/dsviz-nature-colab) に従うか、[Artifacts コンテキスト](https://wandb.ai/stacey/mendeleev/artifacts/val_epoch_preds/val_pred_gawf9z8j/2dcee8fa22863317472b/files/val_epoch_res.table.json) を探索してください。

{{< img src="/images/data_vis/image_classification.png" alt="2つの異なるモデルの予測に基づく真のラベルの分布を比較する。" max-width="90%">}}

### オーディオ

ティンバートランスファーに関する [このレポート](https://wandb.ai/stacey/cshanty/reports/Whale2Song-W-B-Tables-for-Audio--Vmlldzo4NDI3NzM) にてオーディオテーブルと対話します。録音された鯨の歌と、バイオリンやトランペットのような楽器で同じメロディーを合成したものを比較することができます。さらに、[この colab](http://wandb.me/audio-transfer) を使用して自分の曲を録音し、その合成バージョンを W&B で探索できます。

{{< img src="/images/data_vis/audio.png" alt="" max-width="90%">}}

### テキスト

トレーニングデータから生成された出力を参照し、関連するフィールドで動的にグループ化し、モデルバリエーションや実験設定に沿った評価を行います。テキストを Markdown としてレンダリングするか、視覚的な差分モードを使用してテキストを比較します。[このレポート](https://wandb.ai/stacey/nlg/reports/Visualize-Text-Data-Predictions--Vmlldzo1NzcwNzY) でシェイクスピアを生成するための単純な文字ベースの RNN を探索してください。

{{< img src="/images/data_vis/shakesamples.png" alt="隠れ層のサイズを倍増させると、さらに創造的なプロンプトの補完が可能になります。" max-width="90%">}}

### ビデオ

トレーニング中にログされたビデオを表示し、集計してモデルを理解します。ここは、RL エージェントが [副作用を最小限に抑える](https://wandb.ai/stacey/saferlife/artifacts/video/videos_append-spawn/c1f92c6e27fa0725c154/files/video_examples.table.json) ために [SafeLife ベンチマーク](https://wandb.ai/safelife/v1dot2/benchmark) を使用した初期の例です。

{{< img src="/images/data_vis/video.png" alt="数少ない成功したエージェントを簡単に参照する" max-width="90%">}}

### 表形式データ

バージョン管理と重複排除を使用して[表形式データを分割および前処理する方法](https://wandb.ai/dpaiton/splitting-tabular-data/reports/Tabular-Data-Versioning-and-Deduplication-with-Weights-Biases--VmlldzoxNDIzOTA1) に関するレポートを参照します。

{{< img src="/images/data_vis/tabs.png" alt="Tables と Artifacts が連携して、データセットのイテレーションのバージョン管理、ラベル付け、重複排除を行う" max-width="90%">}}

### モデルバリエーションの比較 (セマンティックセグメンテーション)

セマンティックセグメンテーションのための Tables のログ記録と異なるモデルの比較の [インタラクティブなノートブック](https://wandb.me/dsviz-cars-demo) と [ライブ例](https://wandb.ai/stacey/evalserver_answers_2/artifacts/results/eval_Daenerys/c2290abd3d7274f00ad8/files/eval_results.table.json#a57f8e412329727038c2$eval_Ada)。[この Table](https://wandb.ai/stacey/evalserver_answers_2/artifacts/results/eval_Daenerys/c2290abd3d7274f00ad8/files/eval_results.table.json) で独自のクエリを試してください。

{{< img src="/images/data_vis/comparing_model_variants.png" alt="同じテストセットの 2 つのモデル間で最高の予測を見つける" max-width="90%" >}}

### トレーニング期間中の改善の分析

[時間をかけた予測を視覚化](https://wandb.ai/stacey/mnist-viz/reports/Visualize-Predictions-over-Time--Vmlldzo1OTQxMTk) する方法に関する詳細なレポートと、付随する [インタラクティブなノートブック](https://wandb.me/dsviz-mnist-colab)。
---
title: Example tables
description: W&B Tables の例
menu:
  default:
    identifier: ja-guides-core-tables-tables-gallery
    parent: tables
---

以下のセクションでは、テーブル の使用方法について説明します。

### データ を表示する

モデル トレーニング または評価中に、メトリクス とリッチメディアをログに記録し、クラウド または [ホスティングインスタンス]({{< relref path="/guides/hosting" lang="ja" >}}) に同期された永続的なデータベースで結果を可視化します。

{{< img src="/images/data_vis/tables_see_data.png" alt="データ の例を参照し、カウントと分布を確認します" max-width="90%" >}}

たとえば、[写真 データセット のバランスの取れた分割を示す](https://wandb.ai/stacey/mendeleev/artifacts/balanced_data/inat_80-10-10_5K/ab79f01e007113280018/files/data_split.table.json)このテーブル を確認してください。

### データ をインタラクティブ に探索する

テーブル を表示、ソート、フィルタリング、グループ化、結合、およびクエリして、データ とモデル のパフォーマンスを理解します。静的なファイルを参照したり、分析 スクリプト を再実行したりする必要はありません。

{{< img src="/images/data_vis/explore_data.png" alt="オリジナルの曲とその合成バージョンを聴く (音色転送あり)" max-width="90%">}}

たとえば、[スタイルが転送されたオーディオ](https://wandb.ai/stacey/cshanty/reports/Whale2Song-W-B-Tables-for-Audio--Vmlldzo4NDI3NzM)に関するこの report を参照してください。

### モデル バージョン を比較する

さまざまなトレーニング エポック 、データセット 、ハイパーパラメーター の選択、モデル アーキテクチャ などを比較して、結果 を迅速に比較します。

{{< img src="/images/data_vis/compare_model_versions.png" alt="詳細な違いを見る: 左側のモデル は赤い歩道を検出し、右側のモデル は検出していません。" max-width="90%">}}

たとえば、[同じテスト 画像 で 2 つのモデル を比較する](https://wandb.ai/stacey/evalserver_answers_2/artifacts/results/eval_Daenerys/c2290abd3d7274f00ad8/files/eval_results.table.json#b6dae62d4f00d31eeebf$eval_Bob)このテーブル を参照してください。

### すべての詳細を追跡し、全体像を見る

特定のステップ での特定の予測 を可視化するためにズームインします。集計統計を表示したり、エラー のパターンを特定したり、改善の機会を理解したりするためにズームアウトします。このツール は、単一のモデル トレーニング からのステップ 、または異なるモデル バージョン 間での結果 の比較に役立ちます。

{{< img src="/images/data_vis/track_details.png" alt="" >}}

たとえば、[MNIST データセット で 1 回、次に 5 回のエポック 後に結果 を分析する](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json#7dd0cd845c0edb469dec)このサンプルテーブル を参照してください。
## W&B Tables を使用したプロジェクト の例
以下は、W&B Tables を使用する実際の W&B プロジェクト の一部を紹介します。

### 画像分類

[この report](https://wandb.ai/stacey/mendeleev/reports/Visualize-Data-for-Image-Classification--VmlldzozNjE3NjA) を読むか、[この colab](https://wandb.me/dsviz-nature-colab) に従うか、または [Artifacts コンテキスト](https://wandb.ai/stacey/mendeleev/artifacts/val_epoch_preds/val_pred_gawf9z8j/2dcee8fa22863317472b/files/val_epoch_res.table.json) を調べて、CNN が [iNaturalist](https://www.inaturalist.org/pages/developers) の写真から 10 種類の生物 (植物、鳥、昆虫など) を識別する方法を確認してください。

{{< img src="/images/data_vis/image_classification.png" alt="2 つの異なるモデル の予測 における真のラベル の分布を比較します。" max-width="90%">}}

### オーディオ

[この report](https://wandb.ai/stacey/cshanty/reports/Whale2Song-W-B-Tables-for-Audio--Vmlldzo4NDI3NzM) で音色転送に関するオーディオテーブル を操作します。録音されたクジラの歌と、バイオリンやトランペットなどの楽器で同じメロディーを合成した演奏を比較できます。[この colab](http://wandb.me/audio-transfer) を使用して、自分の曲を録音し、W&B で合成バージョンを探索することもできます。

{{< img src="/images/data_vis/audio.png" alt="" max-width="90%">}}

### テキスト

トレーニングデータ または生成された出力からテキストサンプル を参照し、関連フィールドで動的にグループ化し、モデル バリアント または実験 設定全体で評価を調整します。テキストを Markdown としてレンダリングするか、ビジュアル diff モードを使用してテキストを比較します。[この report](https://wandb.ai/stacey/nlg/reports/Visualize-Text-Data-Predictions--Vmlldzo1NzcwNzY) で、シェイクスピアを生成するための単純な文字ベースの RNN を調べてください。

{{< img src="/images/data_vis/shakesamples.png" alt="隠れ層のサイズを 2 倍にすると、よりクリエイティブ なプロンプトの完了が得られます。" max-width="90%">}}

### ビデオ

トレーニング 中にログ に記録されたビデオ を参照および集約して、モデル を理解します。[副作用を最小限に抑える](https://wandb.ai/stacey/saferlife/artifacts/video/videos_append-spawn/c1f92c6e27fa0725c154/files/video_examples.table.json) ことを目指す RL エージェント の [SafeLife ベンチマーク](https://wandb.ai/safelife/v1dot2/benchmark) を使用した初期の例を次に示します。

{{< img src="/images/data_vis/video.png" alt="成功したエージェント を簡単に閲覧できます" max-width="90%">}}

### 表形式 データ

バージョン管理と重複排除を使用して、[表形式 データ を分割および事前処理する方法](https://wandb.ai/dpaiton/splitting-tabular-data/reports/Tabular-Data-Versioning-and-Deduplication-with-Weights-Biases--VmlldzoxNDIzOTA1)に関する report を表示します。

{{< img src="/images/data_vis/tabs.png" alt="Tables と Artifacts は連携して、データセット のイテレーション をバージョン管理、ラベル付け、および重複排除します" max-width="90%">}}

### モデル バリアント の比較 (セマンティックセグメンテーション)

セマンティックセグメンテーション 用のテーブル のログ を記録し、異なるモデル を比較する[インタラクティブ ノートブック](https://wandb.me/dsviz-cars-demo) と [ライブ例](https://wandb.ai/stacey/evalserver_answers_2/artifacts/results/eval_Daenerys/c2290abd3d7274f00ad8/files/eval_results.table.json#a57f8e412329727038c2$eval_Ada)。[このテーブル](https://wandb.ai/stacey/evalserver_answers_2/artifacts/results/eval_Daenerys/c2290abd3d7274f00ad8/files/eval_results.table.json) で独自のクエリ を試してください。

{{< img src="/images/data_vis/comparing_model_variants.png" alt="同じテスト セット で 2 つのモデル 間で最適な予測 を見つけます" max-width="90%" >}}

### トレーニング 時間 における改善の分析

[時間の経過に伴う予測 を可視化する方法](https://wandb.ai/stacey/mnist-viz/reports/Visualize-Predictions-over-Time--Vmlldzo1OTQxMTk)に関する詳細な report と、それに付随する [インタラクティブ ノートブック](https://wandb.me/dsviz-mnist-colab)。

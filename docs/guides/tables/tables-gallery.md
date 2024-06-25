---
description: W&B テーブルの例
displayed_sidebar: default
---


# Tables gallery
このセクションでは、テーブルを使用する方法のいくつかを紹介します。

### データの表示

モデルトレーニングや評価中にメトリクスやリッチメディアをログし、その結果をクラウドと同期した永続的なデータベースで視覚化します。または、[ホスティングインスタンス](https://docs.wandb.ai/guides/hosting)に同期します。

![データのカウントと分布を確認する](/images/data_vis/tables_see_data.png)

たとえば、[写真データセットのバランスの取れた分割](https://wandb.ai/stacey/mendeleev/artifacts/balanced\_data/inat\_80-10-10\_5K/ab79f01e007113280018/files/data\_split.table.json)を示すこのテーブルを確認してください。

### データのインタラクティブな探索

データとモデルのパフォーマンスを理解するために、テーブルを表示、ソート、フィルタリング、グループ化、結合、およびクエリします。静的ファイルを参照する必要もなく、分析スクリプトを再実行する必要もありません。

![元の曲とシンセサイズされたバージョンを聴く（ティンバー転送付き）](/images/data_vis/explore_data.png)

たとえば、[スタイル転送されたオーディオ](https://wandb.ai/stacey/cshanty/reports/Whale2Song-W-B-Tables-for-Audio--Vmlldzo4NDI3NzM)に関するこのレポートを確認してください。

### モデルバージョンの比較

異なるトレーニングエポック、データセット、ハイパーパラメーターの選択、モデルアーキテクチャーなどの結果をすばやく比較します。

![粒度の細かい違いを見る：左のモデルは赤い歩道の一部を検出し、右のモデルは検出しません。](/images/data_vis/compare_model_versions.png)

たとえば、[同じテストイメージ上の2つのモデル](https://wandb.ai/stacey/evalserver\_answers\_2/artifacts/results/eval\_Daenerys/c2290abd3d7274f00ad8/files/eval\_results.table.json#b6dae62d4f00d31eeebf$eval\_Bob)を比較するこのテーブルを確認してください。

### すべての詳細を追跡し、大局を把握する

特定のステップでの具体的な予測を視覚化するためにズームインします。全体の統計を見渡し、エラーパターンを特定し、改善の機会を理解するためにズームアウトします。このツールは、1つのモデルトレーニングのステップを比較する場合や、異なるモデルバージョンの結果を比較する場合に使用できます。

![](/images/data_vis/track_details.png)

たとえば、[MNISTデータセット](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json#7dd0cd845c0edb469dec)において、1エポック後と5エポック後の結果を分析するこの例を参照してください。

## W&B Tablesを使用したプロジェクトの例
以下は、W&B Tablesを使用している実際のW&B Projectsの例をいくつか紹介します。

### 画像分類

[このレポート](https://wandb.ai/stacey/mendeleev/reports/Visualize-Data-for-Image-Classification--VmlldzozNjE3NjA)を読み、[このコラボ](https://wandb.me/dsviz-nature-colab)をフォローするか、[artifactsコンテキスト](https://wandb.ai/stacey/mendeleev/artifacts/val\_epoch\_preds/val\_pred\_gawf9z8j/2dcee8fa22863317472b/files/val\_epoch\_res.table.json)を探索して、CNNが[インナチュラリスト](https://www.inaturalist.org/pages/developers)の写真から10種類の生物（植物、鳥、昆虫など）をどのように識別するかを確認してください。

![2つの異なるモデルの予測における正しいラベルの分布を比較する](/images/data_vis/image_classification.png)

### オーディオ

ティンバー転送に関する[このレポート](https://wandb.ai/stacey/cshanty/reports/Whale2Song-W-B-Tables-for-Audio--Vmlldzo4NDI3NzM)でオーディオテーブルを操作します。録音されたクジラの歌とバイオリンやトランペットなどの楽器でシンセサイズされた同じメロディーのバージョンを比較できます。また、[このコラボ](http://wandb.me/audio-transfer)で自分の曲を録音し、シンセサイズされたバージョンをW&Bで探索できます。

![](/images/data_vis/audio.png)

### テキスト

トレーニングデータからのテキストサンプルや生成された出力を閲覧し、関連フィールドで動的にグループ化し、モデルバリアントや実験設定に合わせて評価を整えます。テキストをMarkdownとしてレンダリングするか、ビジュアルディフモードを使ってテキストを比較します。[このレポート](https://wandb.ai/stacey/nlg/reports/Visualize-Text-Data-Predictions--Vmlldzo1NzcwNzY)でシェイクスピアを生成するシンプルな文字ベースのRNNを探索します。

![隠れ層のサイズを2倍にすると、より創造的なプロンプトの完了が得られます。](@site/static/images/data_vis/shakesamples.png)

### ビデオ

トレーニング中にログされたビデオを閲覧し、集計してモデルを理解します。[SafeLifeベンチマーク](https://wandb.ai/safelife/v1dot2/benchmark)を使用し、RLエージェントが[副作用を最小限に抑える](https://wandb.ai/stacey/saferlife/artifacts/video/videos\_append-spawn/c1f92c6e27fa0725c154/files/video\_examples.table.json)ための初期の例です。

![いくつかの成功したエージェントを簡単に閲覧](/images/data_vis/video.png)

### 表形式データ

バージョン管理と重複排除を使用して、[表形式データを分割および前処理](https://wandb.ai/dpaiton/splitting-tabular-data/reports/Tabular-Data-Versioning-and-Deduplication-with-Weights-Biases--VmlldzoxNDIzOTA1)する方法についてのレポートを確認します。

![TablesとArtifactsは連携して、データセットのバージョン管理、ラベル付け、および重複排除を行います](@site/static/images/data_vis/tabs.png)

### モデルバリエーションの比較（セマンティックセグメンテーション）

セマンティックセグメンテーションと異なるモデルを比較するためのTablesのログの[インタラクティブなノートブック](https://wandb.me/dsviz-cars-demo)および[ライブ例](https://wandb.ai/stacey/evalserver\_answers\_2/artifacts/results/eval\_Daenerys/c2290abd3d7274f00ad8/files/eval\_results.table.json#a57f8e412329727038c2$eval\_Ada)。[このテーブル](https://wandb.ai/stacey/evalserver\_answers\_2/artifacts/results/eval\_Daenerys/c2290abd3d7274f00ad8/files/eval\_results.table.json)で自分のクエリを試してみてください。

![同じテストセット上の2つのモデル間で最高の予測を見つける](/images/data_vis/comparing_model_variants.png)

### トレーニング期間にわたる改善の分析

[時間経過とともに予測を視覚化](https://wandb.ai/stacey/mnist-viz/reports/Visualize-Predictions-over-Time--Vmlldzo1OTQxMTk)する方法に関する詳細なレポートおよび付随する[インタラクティブなノートブック](https://wandb.me/dsviz-mnist-colab)。
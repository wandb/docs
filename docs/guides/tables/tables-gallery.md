---
description: "W&B \u30C6\u30FC\u30D6\u30EB\u306E\u4F8B"
displayed_sidebar: default
---

# Tables gallery
以下のセクションでは、テーブルを使用するいくつかの方法を紹介します。

### データを表示する

モデルのトレーニングや評価中にメトリクスやリッチメディアをログし、その結果をクラウドに同期された永続的なデータベースや[ホスティングインスタンス](https://docs.wandb.ai/guides/hosting)で視覚化します。

![データの例を閲覧し、データのカウントと分布を確認する](/images/data_vis/tables_see_data.png)

例えば、[写真データセットのバランスの取れた分割](https://wandb.ai/stacey/mendeleev/artifacts/balanced\_data/inat\_80-10-10\_5K/ab79f01e007113280018/files/data\_split.table.json)を示すこのテーブルをチェックしてください。

### データをインタラクティブに探索する

データとモデルのパフォーマンスを理解するために、テーブルを表示、ソート、フィルタリング、グループ化、結合、クエリします。静的ファイルを閲覧したり、分析スクリプトを再実行する必要はありません。

![オリジナルの曲とその合成バージョン（ティンバートランスファー）を聴く](/images/data_vis/explore_data.png)

例えば、[スタイル転送されたオーディオ](https://wandb.ai/stacey/cshanty/reports/Whale2Song-W-B-Tables-for-Audio--Vmlldzo4NDI3NzM)に関するこのレポートを参照してください。

### モデルバージョンを比較する

異なるトレーニングエポック、データセット、ハイパーパラメーターの選択、モデルアーキテクチャーなどの結果を迅速に比較します。

![詳細な違いを見る: 左のモデルは赤い歩道を検出し、右のモデルは検出しない。](/images/data_vis/compare_model_versions.png)

例えば、[同じテスト画像で2つのモデルを比較](https://wandb.ai/stacey/evalserver\_answers\_2/artifacts/results/eval\_Daenerys/c2290abd3d7274f00ad8/files/eval\_results.table.json#b6dae62d4f00d31eeebf$eval\_Bob)するこのテーブルを参照してください。

### すべての詳細を追跡し、大局を見る

特定のステップでの特定の予測を視覚化するためにズームインします。集計統計を見て、エラーパターンを特定し、改善の機会を理解するためにズームアウトします。このツールは、単一のモデルトレーニングのステップを比較する場合や、異なるモデルバージョンの結果を比較する場合に機能します。

![](/images/data_vis/track_details.png)

例えば、[MNISTデータセットでの1エポック後と5エポック後の結果](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json#7dd0cd845c0edb469dec)を分析するこの例のテーブルを参照してください。

## W&B Tablesを使用したプロジェクトの例
以下は、W&B Tablesを使用した実際のW&B Projectsのいくつかを紹介します。

### 画像分類

[このレポート](https://wandb.ai/stacey/mendeleev/reports/Visualize-Data-for-Image-Classification--VmlldzozNjE3NjA)を読み、[このcolab](https://wandb.me/dsviz-nature-colab)をフォローし、[artifacts context](https://wandb.ai/stacey/mendeleev/artifacts/val\_epoch\_preds/val\_pred\_gawf9z8j/2dcee8fa22863317472b/files/val\_epoch\_res.table.json)を探索して、CNNが[iNaturalist](https://www.inaturalist.org/pages/developers)の写真から10種類の生物（植物、鳥、昆虫など）を識別する方法を確認してください。

![2つの異なるモデルの予測における真のラベルの分布を比較する。](/images/data_vis/image_classification.png)

### オーディオ

ティンバートランスファーに関する[このレポート](https://wandb.ai/stacey/cshanty/reports/Whale2Song-W-B-Tables-for-Audio--Vmlldzo4NDI3NzM)でオーディオテーブルと対話します。録音されたクジラの歌と、バイオリンやトランペットのような楽器で同じメロディーを合成したバージョンを比較できます。また、自分の曲を録音し、W&Bでその合成バージョンを探索することもできます。[このcolab](http://wandb.me/audio-transfer)を使用してください。

![](/images/data_vis/audio.png)

### テキスト

トレーニングデータや生成された出力のテキストサンプルを閲覧し、関連フィールドで動的にグループ化し、モデルバリアントや実験設定全体で評価を整合させます。テキストをMarkdownとしてレンダリングしたり、ビジュアルディフモードを使用してテキストを比較します。[このレポート](https://wandb.ai/stacey/nlg/reports/Visualize-Text-Data-Predictions--Vmlldzo1NzcwNzY)でシェイクスピアを生成するためのシンプルな文字ベースのRNNを探索してください。

![隠れ層のサイズを倍にすると、より創造的なプロンプトの完了が得られます。](@site/static/images/data_vis/shakesamples.png)

### ビデオ

トレーニング中にログされたビデオを閲覧し、集計してモデルを理解します。RLエージェントが[副作用を最小限に抑える](https://wandb.ai/stacey/saferlife/artifacts/video/videos\_append-spawn/c1f92c6e27fa0725c154/files/video\_examples.table.json)ために[SafeLifeベンチマーク](https://wandb.ai/safelife/v1dot2/benchmark)を使用した初期の例です。

![成功したエージェントを簡単に閲覧する](/images/data_vis/video.png)

### 表形式データ

バージョン管理と重複排除を使用して[表形式データを分割および前処理する方法](https://wandb.ai/dpaiton/splitting-tabular-data/reports/Tabular-Data-Versioning-and-Deduplication-with-Weights-Biases--VmlldzoxNDIzOTA1)に関するレポートを参照してください。

![TablesとArtifactsが連携して、データセットのイテレーションをバージョン管理、ラベル付け、および重複排除します](@site/static/images/data_vis/tabs.png)

### モデルバリアントの比較（セマンティックセグメンテーション）

セマンティックセグメンテーションのためのTablesのログと異なるモデルの比較を行う[インタラクティブなノートブック](https://wandb.me/dsviz-cars-demo)と[ライブ例](https://wandb.ai/stacey/evalserver\_answers\_2/artifacts/results/eval\_Daenerys/c2290abd3d7274f00ad8/files/eval\_results.table.json#a57f8e412329727038c2$eval\_Ada)。[このTable](https://wandb.ai/stacey/evalserver\_answers\_2/artifacts/results/eval\_Daenerys/c2290abd3d7274f00ad8/files/eval\_results.table.json)で自分のクエリを試してみてください。

![同じテストセットで2つのモデルの最高の予測を見つける](/images/data_vis/comparing_model_variants.png)

### トレーニング時間の改善を分析する

[時間経過に伴う予測を視覚化する方法](https://wandb.ai/stacey/mnist-viz/reports/Visualize-Predictions-over-Time--Vmlldzo1OTQxMTk)に関する詳細なレポートと、付随する[インタラクティブなノートブック](https://wandb.me/dsviz-mnist-colab)を参照してください。
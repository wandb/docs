---
title: Tableを可視化して解析する
description: W&B Tablesを可視化して分析しましょう。
menu:
  default:
    identifier: ja-guides-models-tables-visualize-tables
    parent: tables
weight: 2
---

W&B Tables をカスタマイズして、機械学習モデルのパフォーマンスに関する疑問へ答えたり、データを解析したりできます。

インタラクティブにデータを探索し、以下のようなことが可能です：

* モデル・エポック・個々のサンプル間で、変化を正確に比較する
* データ内でより高次なパターンを理解する
* ビジュアルなサンプルを通じて考察を記録・共有する



{{% alert %}}
W&B Tables には次のような振る舞いがあります：
1. **アーティファクトのコンテキストではステートレス**：アーティファクトバージョンと一緒に記録されたTableは、ブラウザウィンドウを閉じるとデフォルト状態にリセットされます
2. **ワークスペースやレポートのコンテキストではステートフル**：シングル run ワークスペース、マルチ run Project ワークスペース、または Report 内のTableで行った変更は保存され、次回もそのまま表示されます

現在の W&B Table のビューを保存する方法については[ビューの保存方法]({{< relref path="#save-your-view" lang="ja" >}})を参照してください。
{{% /alert %}}

## 2つのTableを比較する
[マージドビュー]({{< relref path="#merged-view" lang="ja" >}})や[サイドバイサイドビュー]({{< relref path="#side-by-side-view" lang="ja" >}})で2つのTableを比較できます。下の画像はMNISTデータのTable比較例です。

{{< img src="/images/data_vis/table_comparison.png" alt="Training epoch comparison" max-width="90%" >}}

2つのTableを比較する手順は以下の通りです：

1. W&B Appで自分の Project に移動します。
2. 左側パネルの Artifacts アイコンを選択します。
2. 比較したい Artifact バージョンを選択します。

下の画像では、5エポック学習後の MNIST 検証データに対するモデルの予測例を示しています（[インタラクティブな例はこちら](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json)）。

{{< img src="/images/data_vis/preds_mnist.png" alt="Click on 'predictions' to view the Table" max-width="90%" >}}


3. サイドバーで比較したい2つ目の Artifact バージョンにカーソルを合わせ、**Compare** ボタンが現れたらクリックします。例えば、下の画像では「v4」というバージョンを選択し、同じモデルが学習5エポック後に行ったMNISTの予測と比較しています。

{{< img src="/images/data_vis/preds_2.png" alt="Model prediction comparison" max-width="90%" >}}

### マージドビュー

最初は2つのTableがマージされた状態で表示されます。最初に選んだTableがインデックス0（青色のハイライト）、2つ目がインデックス1（黄色のハイライト）です。[マージドTableのライブ例はこちら](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json#7dd0cd845c0edb469dec)。

{{< img src="/images/data_vis/merged_view.png" alt="Merged view" max-width="90%">}}

マージドビューから、以下の操作が可能です：

* **ジョインキーの選択**：左上のドロップダウンで2つのTableを接合するための列（ジョインキー）を設定できます。一般的には各行ごとに一意な識別子（例：データセット内のファイル名や生成サンプルのインデックス）になります。_どの列も_選択可能ですが、可読性が下がったりクエリーが遅くなったりする場合があります。
* **結合の代わりに連結**：このドロップダウンで「全Tableを連結」を選ぶと、両Tableの全行を1つの大きな Table に _union_ して結合します（列ではなく行単位）。
* **各 Table を明示的に参照**：フィルター表現で 0, 1, \* を使って、どの Table の列かを指定できます
* **数値的な詳細差分のヒストグラム可視化**：任意のセルの値の違いを一目で比較できます

### サイドバイサイドビュー

2つのTableを左右に並べて表示したい場合は、最初のドロップダウンを "Merge Tables: Table" から "List of: Table" に切り替え、「Page size」を設定します。ここでは、最初に選択した Table が左側、2つ目が右側です。「Vertical」チェックボックスをオンにすれば上下での比較も可能です。

{{< img src="/images/data_vis/side_by_side.png" alt="Side-by-side table view" max-width="90%" >}}

* **一目でTableを比較**：どちらのTableにも一括で操作（ソート・フィルター・グループ化など）を適用して違いを素早く発見できます。たとえば「推論ミス」を推測値でグループ化したり、「最も難しいサンプル」を特定したり、「正解ラベルごとの信頼度分布」などを閲覧できます。
* **2つのTableを独立して探索**：気になる側や行だけに集中してスクロールすることもできます



## run 全体で値がどう変化するかを可視化

Tableに記録した値が run を通してどのように変化するか、ステップスライダーを使って簡単に確認できます。スライダーを動かすと異なる step で記録された値に切り替わります。例えば loss や accuracy、その他のメトリクスが各 run 後にどう推移したかを簡単に見られます。

このスライダーは指定したキーで step の値を判断します。デフォルトのスライダーキーは `_step` で、これは W&B が自動的に記録する特別なキーです。`_step` は、`wandb.Run.log()` をコードで呼ぶたびに1つずつ増加する整数です。

W&B Table にステップスライダーを追加するには：

1. Project のワークスペースを開きます。
2. ワークスペース右上の **Add panel** をクリックします。
3. **Query panel** を選択します。
4. クエリーエディタで `runs` を選び、キーボードで **Enter** を押します。
5. 歯車アイコンをクリックし、パネルの設定を表示します。
6. **Render As** セレクタで **Stepper** を選択します。
7. **Stepper Key** に `_step` または [スライダー用に使用するキー]({{< relref path="#custom-step-keys" lang="ja" >}}) を設定します。

下の画像は3つの W&B run と、それぞれが step 295 で記録した値を示すクエリパネル例です。

{{< img src="/images/data_vis/stepper_key.png" alt="Step slider feature">}}

W&B App の UI 上で、複数の step に同じ値が見えることがあります。これは、複数の run が異なる step で同一値を記録している場合や、ある run がすべての step で値をログしなかった場合に起こり得ます。もしある step に値がなければ、W&B はそのキーにおいて最後に記録された値を使います。

### カスタムステップキー

ステップキーは、`epoch` や `global_step` のように、run 内で数値的に管理された任意のメトリックを使うことができます。カスタムステップキーを使う場合、W&B はそのキーの値ごとに run 内の step (`_step`) をマッピングします。

下記のTableは、カスタムステップキー `epoch` が、3つの run（`serene-sponge`、`lively-frog`、`vague-cloud`）で `_step` とどう対応するかを示します。それぞれの行は、各 run で特定の `_step` で `wandb.Run.log()` が呼ばれたときになります。各カラムは、その step で記録された epoch 値（あれば）を示します。スペース節約のため、一部の `_step` が省略されています。

最初に `wandb.Run.log()` が呼ばれた時は、どの run も `epoch` を記録していないので、表では空になります。

| `_step` | vague-cloud (`epoch`) | lively-frog(`epoch`) |  serene-sponge (`epoch`) |
| ------- | ------------- | ----------- | ----------- |
| 1 | | |  |
| 2  |   |   | 1  | 
| 4  |   | 1 | 2  |
| 5  | 1 |   |  |
| 6  |  |   | 3  |
| 8  |  | 2 | 4  |
| 10 |  |   | 5  |
| 12 |  | 3 | 6  |
| 14 |  |   |  7 | 
| 15 | 2  |   |  |
| 16 |  | 4 | 8  | 
| 18 |  |   | 9  |
| 20 | 3 | 5 | 10 |

例えば、スライダーで `epoch=1` を選んだ場合：

* `vague-cloud` は `epoch=1` を見つけて `_step=5` で記録された値を返します
* `lively-frog` は `epoch=1` を見つけて `_step=4` の値を返します
* `serene-sponge` は `epoch=1` を見つけて `_step=2` の値を返します

もしスライダーが `epoch=9` なら：

* `vague-cloud` は `epoch=9` が見つからないため、直前の最新値である `epoch=3` を使い、その `_step=20` の値を返します
* `lively-frog` も `epoch=9` を記録していませんが、最新値は `epoch=5` なので `_step=20` の値を返します
* `serene-sponge` は `epoch=9` を見つけて `_step=18` の値を返します




## アーティファクトを比較する
[時間軸でTableを比較]({{< relref path="#compare-tables-across-time" lang="ja" >}})したり、[モデルバリアントで比較]({{< relref path="#compare-tables-across-model-variants" lang="ja" >}})することも可能です。


### 時系列でTableを比較する
トレーニングの進行に従って各ステップでTableをアーティファクトに記録し、モデルのパフォーマンス変化を分析できます。たとえば、各バリデーションステップの最後や、50エポックごと、任意の頻度でTableを記録できます。サイドバイサイドビューを使えば、モデル予測の変化も直感的に可視化できます。

{{< img src="/images/data_vis/compare_across_time.png" alt="Training progress comparison" max-width="90%" >}}

トレーニング進行中の予測をさらに詳しく可視化する方法は、[予測の推移に関するレポート](https://wandb.ai/stacey/mnist-viz/reports/Visualize-Predictions-over-Time--Vmlldzo1OTQxMTk)や、[インタラクティブなノートブック例](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/datasets-predictions/W%26B_Tables_Quickstart.ipynb?_gl=1*kf20ui*_gcl_au*OTI3ODM1OTcyLjE3MzE0MzU1NjU.*_ga*ODEyMjQ4MjkyLjE3MzE0MzU1NjU.*_ga_JH1SJHJQXJ*MTczMTcwNTMwNS45LjEuMTczMTcwNTM5My4zMy4wLjA.*_ga_GMYDGNGKDT*MTczMTcwNTMwNS44LjEuMTczMTcwNTM5My4wLjAuMA..) もご参照ください。

### モデルバリアント間でTableを比較

2つの異なるモデルで、同じステップで記録した2つのアーティファクトバージョンを比較することで、モデル設定（ハイパーパラメーター・ベースアーキテクチャなど）の違いによるパフォーマンスを簡単に評価できます。

例えば、`baseline` と新しいモデルバリアント `2x_layers_2x_lr` の予測を比較します。このバリアントでは、1層目の畳み込みの数が32→64、2層目が128→256、学習率が0.001→0.002に増えています。[こちらのライブ例](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json#2bb3b1d40aa777496b5d$2x_layers_2x_lr)で、サイドバイサイドビューを使い、1エポック（左タブ）と5エポック（右タブ）後の誤予測のみをフィルタして比較できます。

{{< tabpane text=true >}}
{{% tab header="1 training epoch" value="one_epoch" %}}
{{< img src="/images/data_vis/compare_across_variants.png" alt="Performance comparison" >}}
{{% /tab %}}
{{% tab header="5 training epochs" value="five_epochs" %}}
{{< img src="/images/data_vis/compare_across_variants_after_5_epochs.png" alt="Variant performance comparison" >}}
{{% /tab %}}
{{< /tabpane >}}

## ビューの保存

run ワークスペースや Project ワークスペース、またはレポート内で操作した Tables は、ビューの状態が自動的に保存されます。Tableに何らかの操作を適用した後ブラウザを閉じても、次回そのTableに戻った際、前回の設定のまま表示されます。

{{% alert %}}
Artifacts コンテキストで閲覧した Tables はステートレスなままです。
{{% /alert %}}

特定の状態のTableをワークスペースから保存したい場合は、W&B レポートにエクスポートします。Tableをレポートに追加するには：

1. ワークスペースの可視化パネル右上のケバブアイコン（三点縦アイコン）をクリックします。
2. **Share panel** または **Add to report** を選択します。

{{< img src="/images/data_vis/share_your_view.png" alt="Report sharing options" max-width="90%">}}


## 使用例

次のレポートでは W&B Tables の多様なユースケースを紹介しています：

* [Visualize Predictions Over Time](https://wandb.ai/stacey/mnist-viz/reports/Visualize-Predictions-over-Time--Vmlldzo1OTQxMTk)
* [How to Compare Tables in Workspaces](https://wandb.ai/stacey/xtable/reports/How-to-Compare-Tables-in-Workspaces--Vmlldzo4MTc0MTA)
* [Image & Classification Models](https://wandb.ai/stacey/mendeleev/reports/Tables-Tutorial-Visualize-Data-for-Image-Classification--VmlldzozNjE3NjA)
* [Text & Generative Language Models](https://wandb.ai/stacey/nlg/reports/Tables-Tutorial-Visualize-Text-Data-Predictions---Vmlldzo1NzcwNzY)
* [Named Entity Recognition](https://wandb.ai/stacey/ner_spacy/reports/Named-Entity-Recognition--Vmlldzo3MDE3NzQ)
* [AlphaFold Proteins](https://wandb.ai/wandb/examples/reports/AlphaFold-ed-Proteins-in-W-B-Tables--Vmlldzo4ODc0MDc)
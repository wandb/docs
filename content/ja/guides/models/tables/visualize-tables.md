---
title: テーブルを可視化して分析する
description: W&B テーブルを可視化して分析します。
menu:
  default:
    identifier: visualize-tables
    parent: tables
weight: 2
---

W&B Tables をカスタマイズして、機械学習モデルのパフォーマンスに関する疑問に答えたり、データを分析したりできます。

データをインタラクティブに探索して、次のことが可能です:

* モデル、エポック、または個々のサンプル間の違いを正確に比較する
* データ内の高次のパターンを理解する
* 視覚的なサンプルとともにインサイトを捉え、共有する


{{% alert %}}
W&B Tables には次のような挙動があります:
1. **アーティファクトコンテキストではステートレス**: アーティファクトバージョンと一緒にログされたテーブルは、ブラウザウィンドウを閉じるとデフォルトの状態にリセットされます
2. **ワークスペースやレポートコンテキストではステートフル**: シングル run ワークスペース、マルチ run プロジェクトワークスペース、または Reports 内でテーブルに加えた変更は保持されます。

現在の W&B Table のビューを保存する方法については、[ビューの保存方法]({{< relref "#save-your-view" >}})をご覧ください。
{{% /alert %}}

## 2つのテーブルを比較する
[マージビュー]({{< relref "#merged-view" >}})や[サイドバイサイドビュー]({{< relref "#side-by-side-view" >}})で2つのテーブルを比較できます。下の画像は、MNIST データでのテーブル比較例です。

{{< img src="/images/data_vis/table_comparison.png" alt="トレーニングエポック比較" max-width="90%" >}}

2つのテーブルを比較するには、次の手順に従ってください:

1. W&B App でプロジェクトに移動します。
2. 左パネルの Artifacts アイコンを選択します。
2. アーティファクトバージョンを選択します。

以下の画像では、モデルの予測を MNIST のバリデーションデータに対して、5つのエポックごとに表示しています（[インタラクティブな例はこちら](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json)）。

{{< img src="/images/data_vis/preds_mnist.png" alt="「predictions」をクリックして Table を表示" max-width="90%" >}}

3. サイドバーで比較したい2つ目のアーティファクトバージョンの上にカーソルを合わせ、**Compare** をクリックします。例えば、下の画像では「v4」とラベル付けされたバージョンを選択し、同じモデルがトレーニング5エポック後に行った MNIST 予測と比較しています。

{{< img src="/images/data_vis/preds_2.png" alt="モデル予測の比較" max-width="90%" >}}

### マージビュー

最初は両方のテーブルがマージされた状態で表示されます。最初に選択したテーブルはインデックス0（青色のハイライト）、2番目はインデックス1（黄色のハイライト）となります。[マージされたテーブルのライブ例はこちら](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json#7dd0cd845c0edb469dec)。

{{< img src="/images/data_vis/merged_view.png" alt="マージビュー" max-width="90%">}}

マージビューでは以下のことができます:

* **結合キーの選択**: 左上のドロップダウンで結合に使うカラムを選択できます。通常は、データセット内の特定サンプルのファイル名や、生成サンプルのインデックスのような各行の一意識別子が使用されます。なお、_どの_ カラムも結合キーとして選べますが、読みにくいテーブルとなったりクエリが遅くなることがあります。
* **結合の代わりに連結**: このドロップダウンで「全テーブルを連結する（concatenating all tables）」を選ぶと、両テーブルのすべての行を1つの大きな Table に結合できます（カラム単位の結合ではありません）。
* **各 Table を明示的に参照**: フィルタ式で 0, 1, \* を使うことで、片方または両方のテーブルの特定カラムを直接指定できます。
* **数値の差分をヒストグラムで可視化**: 各セルの値の違いを一目で比較できます。

### サイドバイサイドビュー

2つのテーブルを並べて表示するには、左上のドロップダウンで「Merge Tables: Table」を「List of: Table」に変更し、「Page size」も適宜更新します。ここで1番目の Table は左、2番目は右に配置されます。また、「Vertical」チェックボックスをオンにすることで縦に並べて比較もできます。

{{< img src="/images/data_vis/side_by_side.png" alt="サイドバイサイド テーブルビュー" max-width="90%" >}}

* **テーブルを一目で比較**: 両テーブルにソート・フィルタ・グループ化操作などを同時に適用でき、違いや変化をすばやく確認できます。たとえば、予測ミスを仮説ごとにグループ化、最も難しいネガティブサンプル、ラベルごとの信頼度分布の比較などが可能です。
* **2つのテーブルを独立して探索**: 関心のある側や行をスクロールして、じっくり比較できます。

## run で値がどのように変化するかを可視化

Table にログした値が、run 全体でどのように変化しているかをステップスライダーで確認できます。スライダーを動かすことで、異なるステップに記録された値を表示できます。たとえば、損失値や精度などのメトリクスが各 run の後でどのように移り変わるかを見ることができます。

このスライダーは、どのカラムをステップ値として扱うかをキーで決定します。デフォルトのキーは `_step` で、これは W&B が自動でログしてくれる特別なキーです。`_step` は、`wandb.Run.log()` を実行するたびに1ずつ増える整数です。

W&B Table にステップスライダーを追加する方法:

1. プロジェクトの workspace に移動します。
2. workspace 右上の **Add panel** をクリックします。
3. **Query panel** を選択します。
4. クエリエディター内で `runs` を選び、キーボードの **Enter** を押します。
5. ギアアイコン（⚙）をクリックしてパネルの設定を表示します。
6. **Render As** セレクターを **Stepper** に設定します。
7. **Stepper Key** を `_step` もしくは[単位として使うキー]({{< relref "#custom-step-keys" >}})に設定します。

次の画像は、3つの W&B run および各 run でステップ295に記録された値を表示するクエリパネル例です。

{{< img src="/images/data_vis/stepper_key.png" alt="ステップスライダー機能">}}

W&B App の UI では、複数ステップで同じ値が重複表示されることがあります。複数 run が異なるステップで同じ値をログした場合や、run がすべてのステップで値をログしない場合に発生します。あるステップで値が欠損していると、W&B は直前にログされた値をスライダーキーとして利用します。

### カスタムステップキー

ステップキーは任意の数値メトリクス（例: `epoch` や `global_step` など）を指定して、run でカスタムステップキーとしてログできます。カスタムキーを使うと、そのキーの各値を run 内のステップ（`_step`）にマッピングできます。

この表はカスタムステップキー `epoch` が 3つの異なる run: `serene-sponge`, `lively-frog`, `vague-cloud` の `_step` とどうマッピングされるかを示しています。各行は run 内のある `_step` で `wandb.Run.log()` が呼ばれたことを示し、カラムにはそのときにログされた `epoch` の値（あれば）が表示されています。一部の `_step` 値はスペース節約のため省略しています。

最初に `wandb.Run.log()` が呼ばれた際、どの run でも `epoch` の値はログされていないため、`epoch` カラムは空欄です。

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

例えばスライダーを `epoch = 1` にした場合:

* `vague-cloud` は `epoch = 1` を見つけ、`_step = 5`の値を返します
* `lively-frog` は `epoch = 1` を見つけ、`_step = 4`の値を返します
* `serene-sponge` は `epoch = 1` を見つけ、`_step = 2`の値を返します

スライダーを `epoch = 9` にした場合:

* `vague-cloud` は `epoch = 9` をログしていないので、直前の `epoch = 3` の値（`_step = 20`）を返します
* `lively-frog` も `epoch = 9` をログしておらず、直前の `epoch = 5` の値（`_step = 20`）を返します
* `serene-sponge` は `epoch = 9` を見つけ、`_step = 18`の値を返します

## Artifacts の比較
[時間をまたいだテーブル比較]({{< relref "#compare-tables-across-time" >}})や、[モデルバリアントの比較]({{< relref "#compare-tables-across-model-variants" >}})も可能です。

### 時系列でテーブルを比較する

トレーニングの主要ステップごとに Artifacts にテーブルをログして、トレーニング時間に沿ったモデルのパフォーマンスを分析しましょう。例えば、すべてのバリデーションステップの最後や、50エポックごとなど、適切な頻度でテーブルを記録し、サイドバイサイドビューでモデル予測の変化を可視化できます。

{{< img src="/images/data_vis/compare_across_time.png" alt="トレーニング進捗の比較" max-width="90%" >}}

トレーニング時間とともに予測を可視化する詳細なチュートリアルは [predictions over time report](https://wandb.ai/stacey/mnist-viz/reports/Visualize-Predictions-over-Time--Vmlldzo1OTQxMTk) や [ノートブック例](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/datasets-predictions/W%26B_Tables_Quickstart.ipynb?_gl=1*kf20ui*_gcl_au*OTI3ODM1OTcyLjE3MzE0MzU1NjU.*_ga*ODEyMjQ4MjkyLjE3MzE0MzU1NjU.*_ga_JH1SJHJQXJ*MTczMTcwNTMwNS45LjEuMTczMTcwNTM5My4zMy4wLjA.*_ga_GMYDGNGKDT*MTczMTcwNTMwNS44LjEuMTczMTcwNTM5My4wLjAuMA..) を参照してください。

### モデルバリアントをまたいだテーブル比較

2つの異なるモデルで、同じステップでログされた2つのアーティファクトバージョンを比較し、ハイパーパラメータやベースアーキテクチャーの違いによるモデルパフォーマンスを分析できます。

例えば `baseline` と新しいモデルバリアント `2x_layers_2x_lr`（1層目の畳み込み層が32→64、2層目が128→256、学習率が0.001→0.002）の予測を比較します。[このライブ例](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json#2bb3b1d40aa777496b5d$2x_layers_2x_lr)ではサイドバイサイドビューを利用し、1エポック後（左）と5エポック後（右）の誤予測のみを絞り込んで閲覧できます。

{{< tabpane text=true >}}
{{% tab header="1 training epoch" value="one_epoch" %}}
{{< img src="/images/data_vis/compare_across_variants.png" alt="パフォーマンス比較" >}}
{{% /tab %}}
{{% tab header="5 training epochs" value="five_epochs" %}}
{{< img src="/images/data_vis/compare_across_variants_after_5_epochs.png" alt="バリアントパフォーマンス比較" >}}
{{% /tab %}}
{{< /tabpane >}}

## ビューの保存

run ワークスペース、プロジェクトワークスペース、またはレポート内で操作した Tables は自動的にビューの状態が保存されます。操作後にブラウザを閉じても、次にテーブルを開くときには最後に見た設定のまま表示されます。

{{% alert %}}
Artifacts コンテキストで操作した Tables はステートレスのままです。
{{% /alert %}}

ワークスペースから特定の状態のテーブルを保存したい場合、W&B Report へエクスポートできます。テーブルをレポートにエクスポートする手順は次の通りです:

1. ワークスペースの可視化パネル右上のケバブアイコン（三点縦ドット）をクリックします。
2. **Share panel** または **Add to report** を選択します。

{{< img src="/images/data_vis/share_your_view.png" alt="レポート共有オプション" max-width="90%">}}

## 例

W&B Tables の様々なユースケースを紹介する Reports:

* [Visualize Predictions Over Time](https://wandb.ai/stacey/mnist-viz/reports/Visualize-Predictions-over-Time--Vmlldzo1OTQxMTk)
* [How to Compare Tables in Workspaces](https://wandb.ai/stacey/xtable/reports/How-to-Compare-Tables-in-Workspaces--Vmlldzo4MTc0MTA)
* [Image & Classification Models](https://wandb.ai/stacey/mendeleev/reports/Tables-Tutorial-Visualize-Data-for-Image-Classification--VmlldzozNjE3NjA)
* [Text & Generative Language Models](https://wandb.ai/stacey/nlg/reports/Tables-Tutorial-Visualize-Text-Data-Predictions---Vmlldzo1NzcwNzY)
* [Named Entity Recognition](https://wandb.ai/stacey/ner_spacy/reports/Named-Entity-Recognition--Vmlldzo3MDE3NzQ)
* [AlphaFold Proteins](https://wandb.ai/wandb/examples/reports/AlphaFold-ed-Proteins-in-W-B-Tables--Vmlldzo4ODc0MDc)
---
title: テーブルを可視化・分析する
description: W&B テーブルを可視化して分析する。
menu:
  default:
    identifier: ja-guides-models-tables-visualize-tables
    parent: tables
weight: 2
---

W&B Tables をカスタマイズして、機械学習 モデルの性能に関する疑問に答えたり、データ を分析したり、さらに多くのことができます。 
データ を対話的に探索して、次のことができます:
* モデル、エポック、または個々のサンプル間の変化を正確に比較する
* データ 内の高次のパターンを理解する
* 視覚的なサンプルを使って洞察を記録し、伝える

{{% alert %}}
W&B Tables には次のような振る舞い があります:
1. ステートレス（Artifact コンテキスト）: Artifact バージョンと一緒にログ された任意のテーブルは、ブラウザ ウィンドウを閉じるとデフォルト状態にリセットされます
2. ステートフル（Workspace または Report コンテキスト）: 単一 Run の Workspace、複数 Run の Project Workspace、または Report にあるテーブルに加えた変更は保持されます。

現在の W&B Table のビューを保存する方法は、[ビューを保存]({{< relref path="#save-your-view" lang="ja" >}}) を参照してください。
{{% /alert %}}

## 2 つのテーブルを比較する
[マージドビュー]({{< relref path="#merged-view" lang="ja" >}}) または [並列ビュー]({{< relref path="#side-by-side-view" lang="ja" >}}) で 2 つのテーブルを比較します。たとえば、以下の画像は MNIST データ のテーブル比較を示しています。

{{< img src="/images/data_vis/table_comparison.png" alt="トレーニング エポックの比較" max-width="90%" >}}

2 つのテーブルを比較する手順:

1. W&B App で自分の Project に移動します。
2. 左のパネルで Artifacts アイコンを選択します。
2. Artifact バージョンを選択します。 

次の画像は、5 つのエポックそれぞれの後の MNIST 検証データ に対するモデル の予測を示しています（[インタラクティブな例はこちら](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json)）。

{{< img src="/images/data_vis/preds_mnist.png" alt="「predictions」をクリックして Table を表示" max-width="90%" >}}

3. サイドバーで比較したい 2 つ目の Artifact バージョンにカーソルを合わせ、表示される **Compare** をクリックします。例えば、下の画像では「v4」とラベル付けされたバージョンを選び、同じモデル がトレーニング 5 エポック後に行った MNIST の予測と比較しています。 

{{< img src="/images/data_vis/preds_2.png" alt="モデル 予測の比較" max-width="90%" >}}

### マージドビュー
最初は 2 つのテーブルがマージされて表示されます。最初に選んだテーブルはインデックス 0 で青のハイライト、2 つ目はインデックス 1 で黄色のハイライトです。[マージしたテーブルのライブ例はこちら](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json#7dd0cd845c0edb469dec)。

{{< img src="/images/data_vis/merged_view.png" alt="マージドビュー" max-width="90%">}}

マージドビューでは、次のことができます

* 結合キーを選ぶ: 左上のドロップダウンで、2 つのテーブルを結合する際のキーに使う列を設定します。通常は各行の一意な識別子（データセット 内の特定サンプルのファイル名や、生成サンプルの増分インデックスなど）です。現在は _任意_ の列を選べるため、判読しづらいテーブルやクエリの低速化を招く可能性があります。
* 結合ではなく連結: このドロップダウンで "concatenating all tables" を選ぶと、列同士で結合するのではなく、両テーブルの全行を 1 つの大きな Table にユニオンします
* 各 Table を明示的に参照: フィルター式で 0、1、そして \* を使い、片方または両方のテーブル インスタンス内の列を明示的に指定します
* 数値の詳細な差分をヒストグラムで可視化: 任意のセルの値 をひと目で比較できます

### 並列ビュー
2 つのテーブルを並べて表示するには、最初のドロップダウンを "Merge Tables: Table" から "List of: Table" に変更し、続いて "Page size" を調整します。左に最初に選んだ Table、右に 2 つ目の Table が表示されます。また、"Vertical" チェックボックスをクリックすると縦方向の比較もできます。

{{< img src="/images/data_vis/side_by_side.png" alt="テーブルの並列ビュー" max-width="90%" >}}

* テーブルをひと目で比較: 2 つのテーブルに対して並行して操作（ソート、フィルター、グループ化）を適用し、変化や差分をすばやく見つけられます。例えば、推測ごとにグループ化した誤予測、全体で最も難しいネガティブ、真のラベル別の信頼度スコア分布などを確認できます。
* 2 つのテーブルを独立して探索: 気になる側／行にフォーカスしながらスクロールできます

## Run 全体で値 がどのように変化するかを可視化する
ステップ スライダーを使って、テーブルにログ した値 が Run 全体でどのように変化するかを表示します。スライダーを動かすと、異なるステップでログ された値 を確認できます。例えば、各 Run の後に loss、accuracy、その他のメトリクス がどのように変化したかを見られます。 

スライダーはステップ値 を決めるためのキーを使います。デフォルトのキーは `_step` で、W&B が自動的にログ する特別なキーです。`_step` は、コード内で `wandb.Run.log()` を呼ぶたびに 1 ずつ増える整数です。

W&B Table にステップ スライダーを追加するには:

1. Project の Workspace に移動します。
2. Workspace 右上の **Add panel** をクリックします。
3. **Query panel** を選択します。
4. クエリ式エディタで `runs` を選び、キーボードの **Enter** を押します。
5. 歯車アイコンをクリックしてパネルの設定 を表示します。
6. **Render As** を **Stepper** に設定します。
7. **Stepper Key** を `_step` か、ステップ スライダーの単位として使う [キー]({{< relref path="#custom-step-keys" lang="ja" >}}) に設定します。

次の画像は、3 つの W&B Runs と、ステップ 295 にログ された値 を表示する Query パネルです。

{{< img src="/images/data_vis/stepper_key.png" alt="ステップ スライダー機能">}}

W&B App の UI では、複数のステップで重複する値 が見える場合があります。これは、複数の Run が異なるステップで同じ値 をログ した場合や、Run が毎ステップ値 をログ していない場合に発生します。特定のステップで値 が欠けている場合、W&B はスライダー キーとして最後にログ された値 を使用します。

### カスタム ステップ キー
ステップ キーには、`epoch` や `global_step` など、Run でログ する任意の数値メトリクス を使えます。カスタム ステップ キーを使うと、W&B はそのキーの各値 を Run 内のステップ（`_step`）にマッピングします。

次の表は、カスタム ステップ キー `epoch` が、3 つの異なる Run（`serene-sponge`、`lively-frog`、`vague-cloud`）で `_step` にどのようにマップされるかを示します。各行は、Run の特定の `_step` における `wandb.Run.log()` の呼び出しを表します。各列は、そのステップでログ された（あれば）対応する epoch の値 を示します。紙幅の都合で一部の `_step` は省略しています。

`wandb.Run.log()` が最初に呼ばれたとき、どの Run も `epoch` の値 をログ していなかったため、表の `epoch` は空欄になっています。 

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

スライダーを `epoch = 1` に設定すると、次のようになります:

* `vague-cloud` は `epoch = 1` を見つけ、`_step = 5` でログ された値 を返します
* `lively-frog` は `epoch = 1` を見つけ、`_step = 4` でログ された値 を返します
* `serene-sponge` は `epoch = 1` を見つけ、`_step = 2` でログ された値 を返します

スライダーを `epoch = 9` に設定すると:

* `vague-cloud` は `epoch = 9` をログ していないため、直前の最新値 `epoch = 3` を使用し、`_step = 20` でログ された値 を返します
* `lively-frog` も `epoch = 9` をログ していませんが、直前の最新値 は `epoch = 5` なので、`_step = 20` でログ された値 を返します
* `serene-sponge` は `epoch = 9` を見つけ、`_step = 18` でログ された値 を返します

## Artifacts を比較する
[時間をまたいでテーブルを比較]({{< relref path="#compare-tables-across-time" lang="ja" >}})したり、[モデル バリアント間で比較]({{< relref path="#compare-tables-across-model-variants" lang="ja" >}})したりもできます。 

### 時間をまたいでテーブルを比較する
トレーニング の各意味のあるステップごとに Artifact にテーブルをログ して、トレーニング 時間にわたるモデル 性能を分析します。例えば、各検証ステップの終わりに、トレーニング の 50 エポックごとに、またはあなたのパイプライン に適した任意の頻度でテーブルをログ できます。並列ビューを使って、モデル の予測の変化を可視化します。

{{< img src="/images/data_vis/compare_across_time.png" alt="トレーニング 進捗の比較" max-width="90%" >}}

トレーニング 時間にわたる予測の可視化についてのより詳しい手順は、[predictions over time report](https://wandb.ai/stacey/mnist-viz/reports/Visualize-Predictions-over-Time--Vmlldzo1OTQxMTk) と、このインタラクティブな [notebook example](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/datasets-predictions/W%26B_Tables_Quickstart.ipynb?_gl=1*kf20ui*_gcl_au*OTI3ODM1OTcyLjE3MzE0MzU1NjU.*_ga*ODEyMjQ4MjkyLjE3MzE0MzU1NjU.*_ga_JH1SJHJQXJ*MTczMTcwNTMwNS45LjEuMTczMTcwNTM5My4zMy4wLjA.*_ga_GMYDGNGKDT*MTczMTcwNTMwNS44LjEuMTczMTcwNTM5My4wLjAuMA..) を参照してください。

### モデル バリアント間でテーブルを比較する
2 つの異なるモデル のために同じステップでログ された 2 つの Artifact バージョンを比較して、異なる設定（ハイパーパラメーター、ベース アーキテクチャー など）にわたるモデル 性能を分析します。

例えば、`baseline` と新しいモデル バリアント `2x_layers_2x_lr` の予測を比較します。ここでは、最初の畳み込み層を 32 から 64 に、2 番目を 128 から 256 に、学習率を 0.001 から 0.002 に倍増させています。[このライブ例](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json#2bb3b1d40aa777496b5d$2x_layers_2x_lr) では、並列ビューを使い、1（左タブ）対 5（右タブ）トレーニング エポック後の誤予測に絞り込んで比較します。

{{< tabpane text=true >}}
{{% tab header="トレーニング 1 エポック" value="one_epoch" %}}
{{< img src="/images/data_vis/compare_across_variants.png" alt="性能比較" >}}
{{% /tab %}}
{{% tab header="トレーニング 5 エポック" value="five_epochs" %}}
{{< img src="/images/data_vis/compare_across_variants_after_5_epochs.png" alt="バリアントの性能比較" >}}
{{% /tab %}}
{{< /tabpane >}}

## ビューを保存する
Run Workspace、Project Workspace、または Report で操作した Tables は、自動的にビューステートを保存します。テーブル操作を適用した後にブラウザを閉じても、次にそのテーブルに移動した際に最後に表示した設定 が保持されます。 

{{% alert %}}
Artifact コンテキストで操作した Tables はステートレスのままです。
{{% /alert %}}

Workspace での特定の状態のテーブルを保存するには、W&B Report にエクスポートします。テーブルを Report にエクスポートするには:
1. Workspace の可視化 パネル右上にあるケバブ アイコン（縦三点）を選択します。
2. **Share panel** または **Add to report** を選択します。

{{< img src="/images/data_vis/share_your_view.png" alt="Report 共有オプション" max-width="90%">}}

## 例
これらの Reports では、W&B Tables のさまざまなユースケース を取り上げています:

* [時間をまたいだ予測の可視化](https://wandb.ai/stacey/mnist-viz/reports/Visualize-Predictions-over-Time--Vmlldzo1OTQxMTk)
* [Workspace でテーブルを比較する方法](https://wandb.ai/stacey/xtable/reports/How-to-Compare-Tables-in-Workspaces--Vmlldzo4MTc0MTA)
* [画像 & 分類モデル](https://wandb.ai/stacey/mendeleev/reports/Tables-Tutorial-Visualize-Data-for-Image-Classification--VmlldzozNjE3NjA)
* [テキスト & 生成言語モデル](https://wandb.ai/stacey/nlg/reports/Tables-Tutorial-Visualize-Text-Data-Predictions---Vmlldzo1NzcwNzY)
* [固有表現抽出](https://wandb.ai/stacey/ner_spacy/reports/Named-Entity-Recognition--Vmlldzo3MDE3NzQ)
* [AlphaFold タンパク質](https://wandb.ai/wandb/examples/reports/AlphaFold-ed-Proteins-in-W-B-Tables--Vmlldzo4ODc0MDc)
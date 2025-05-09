---
title: テーブルを視覚化して分析する
description: W&B テーブルを視覚化して分析する。
menu:
  default:
    identifier: ja-guides-models-tables-visualize-tables
    parent: tables
weight: 2
---

W&B Tables をカスタマイズして、機械学習モデルの性能に関する質問に答え、データを分析するなどの操作を行いましょう。

データを対話的に探索して、以下のことができます。

* モデル、エポック、または個々の例を精密に比較
* データ内の高次のパターンを理解
* ビジュアルサンプルで洞察をキャプチャして伝える

{{% alert %}}
W&B Tables には以下の振る舞いがあります:
1. **アーティファクトコンテキストでのステートレス性**: アーティファクトバージョンと共にログされたテーブルは、ブラウザウィンドウを閉じるとデフォルト状態にリセットされます。
2. **ワークスペースまたはレポートコンテキストでのステートフル性**: シングルrunのワークスペース、マルチrunプロジェクトワークスペース、またはレポート内でテーブルに加えた変更は永続化されます。

現在の W&B Table ビューを保存する方法については、[ビューを保存する]({{< relref path="#save-your-view" lang="ja" >}})を参照してください。
{{% /alert %}}

## 2つのテーブルを表示する方法
[マージされたビュー]({{< relref path="#merged-view" lang="ja" >}})または[並列ビュー]({{< relref path="#side-by-side-view" lang="ja" >}})で2つのテーブルを比較します。以下の画像は、MNISTデータのテーブル比較を示しています。

{{< img src="/images/data_vis/table_comparison.png" alt="左: 1 エポックのトレーニング後の誤り, 右: 5 エポック後の誤り" max-width="90%" >}}

2つのテーブルを比較する手順:

1. W&B Appでプロジェクトに移動します。
2. 左のパネルでアーティファクトのアイコンを選択します。
3. アーティファクトバージョンを選択します。

以下の画像では、5 エポック後の各MNIST検証データにおけるモデルの予測を示しています（[対話型の例はこちら](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json)を参照）。

{{< img src="/images/data_vis/preds_mnist.png" alt="テーブルを表示するには「predictions」をクリックします" max-width="90%" >}}

3. サイドバー内で比較したい2つ目のアーティファクトバージョンにカーソルを合わせ、表示されたら**Compare**をクリックします。例えば、以下の画像では、同じモデルによる5 エポック後のMNIST予測と比較するために「v4」とラベル付けされたバージョンを選択しています。

{{< img src="/images/data_vis/preds_2.png" alt="1 エポックのトレーニング後のモデル予測（v0、ここに表示）と5 エポック後（v4）の比較準備中" max-width="90%" >}}

### マージされたビュー

最初は、両方のテーブルが一緒にマージされた状態が表示されます。選択した最初のテーブルはインデックス0で青色のハイライトがあり、2つ目のテーブルはインデックス1で黄色のハイライトがあります。[ここでマージされたテーブルのライブ例を表示](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json#7dd0cd845c0edb469dec)することができます。

{{< img src="/images/data_vis/merged_view.png" alt="マージされたビューでは、数値列はデフォルトでヒストグラムとして表示されます" max-width="90%">}}

マージされたビューから、以下の操作を行うことができます。

* **結合キーを選択**: 左上のドロップダウンを使って、2つのテーブルに対する結合キーとして使用する列を設定します。通常、これは各行の一意の識別子であり、データセット内の特定の例のファイル名や生成されたサンプルのインクリメントインデックスなどです。_任意の_ 列を選択することもできますが、これにより判読不可能なテーブルや遅いクエリが発生する可能性があります。
* **結合の代わりに連結**: このドロップダウンで「すべてのテーブルを連結する」を選択して、両方のテーブルのすべての行を1つの大きなテーブルに_結合して_ 列を横断するようにします。
* **各テーブルを明示的に参照**: フィルター式で0、1、\*を使用して、1つまたは両方のテーブルインスタンスの列を明示的に指定します。
* **ヒストグラムとして詳細な数値差分を可視化**: 任意のセルの値を一目で比較します。

### 並列ビュー

2つのテーブルを並べて表示するには、最初のドロップダウンを「Merge Tables: Table」から「List of: Table」に変更し、「Page size」をそれに応じて更新します。ここで選択された最初のテーブルは左側にあり、2つ目のテーブルは右側にあります。さらに、「Vertical」チェックボックスをクリックして、これらのテーブルを縦に比較することもできます。

{{< img src="/images/data_vis/side_by_side.png" alt="並列ビューでは、テーブルの行はそれぞれ独立しています。" max-width="90%" >}}

* **テーブルを一目で比較**: 両方のテーブルに一緒に操作（並べ替え、フィルター、グループ化）を適用し、すばやく変更や違いを特定できます。たとえば、誤った予測を予測値ごとにグループ化したり、最も難しいネガティブを、真のラベルごとの信頼度スコア分布などを表示できます。
* **2つのテーブルを独立して探索**: 関心のあるサイド/行をスクロールしてフォーカスします。

## アーティファクトを比較
[テーブルを時間で比較]({{< relref path="#compare-tables-across-time" lang="ja" >}})したり、[モデルバリアントを比較]({{< relref path="#compare-tables-across-model-variants" lang="ja" >}})することもできます。

### テーブルを時間で比較
各トレーニングの意味のあるステップに対してアーティファクトにテーブルを記録し、トレーニング時間を通じてモデルの性能を分析します。たとえば、各検証ステップの終了時、毎50エポックのトレーニング後、またはパイプラインに意味を持つ任意の頻度でテーブルを記録できます。モデル予測の変更を可視化するために、並列ビューを使用します。

{{< img src="/images/data_vis/compare_across_time.png" alt="各ラベルについて、5 エポックのトレーニング後（右）は1 エポック後（左）よりも誤りが少ない" max-width="90%" >}}

トレーニング時間を通じて予測を可視化するより詳細なウォークスルーについては、[このレポート](https://wandb.ai/stacey/mnist-viz/reports/Visualize-Predictions-over-Time--Vmlldzo1OTQxMTk)およびこの対話型[ノートブックの例](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/datasets-predictions/W%26B_Tables_Quickstart.ipynb?_gl=1*kf20ui*_gcl_au*OTI3ODM1OTcyLjE3MzE0MzU1NjU.*_ga*ODEyMjQ4MjkyLjE3MzE0MzU1NjU.*_ga_JH1SJHJQXJ*MTczMTcwNTMwNS45LjEuMTczMTcwNTM5My4zMy4wLjA.*_ga_GMYDGNGKDT*MTczMTcwNTMwNS44LjEuMTczMTcwNTM5My4wLjAuMA..)を参照してください。

### モデルバリアントを超えてテーブルを比較

2つの異なるモデルで同じステップでログされた2つのアーティファクトバージョンを比較して、異なる設定（ハイパーパラメーター、基本アーキテクチャーなど）全体でのモデル性能を分析します。

たとえば、`baseline`と新しいモデルバリアント`2x_layers_2x_lr`の予測を比較します。この場合、最初の畳み込み層が32から64に、2番目が128から256に、学習率が0.001から0.002に倍増します。[このライブ例](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json#2bb3b1d40aa777496b5d$2x_layers_2x_lr)から、並列ビューを使用して、1エポック後（左タブ）と5エポック後（右タブ）の誤った予測にフィルターをかけます。

{{< tabpane text=true >}}
{{% tab header="1 トレーニングエポック" value="one_epoch" %}}
{{< img src="/images/data_vis/compare_across_variants.png" alt="1 エポック後、パフォーマンスは混在しています: 一部のクラスでは精度が向上し、他のクラスでは悪化しています。" >}}
{{% /tab %}}
{{% tab header="5 トレーニングエポック" value="five_epochs" %}}
{{< img src="/images/data_vis/compare_across_variants_after_5_epochs.png" alt="5 エポック後、「ダブル」バリアントはベースラインに追いついています。" >}}
{{% /tab %}}
{{< /tabpane >}}

## ビューを保存

runワークスペース、プロジェクトワークスペース、またはレポート内で操作したテーブルは、ビュー状態を自動的に保存します。テーブル操作を適用し、その後ブラウザを閉じても、次にテーブルに戻ったときに最後に表示された設定が保持されます。

{{% alert %}}
アーティファクトコンテキストで操作したテーブルはステートレスのままです。
{{% /alert %}}

特定の状態でワークスペースからテーブルを保存するには、W&B Report にエクスポートします。レポートにテーブルをエクスポートするには:
1. ワークスペースの可視化パネルの右上隅にあるケバブアイコン（縦に並んだ3つの点）を選択します。
2. **Share panel** または **Add to report** を選択します。

{{< img src="/images/data_vis/share_your_view.png" alt="Share panel は新しいレポートを作成し、Add to report は既存のレポートに追加できます。" max-width="90%">}}


## 例

これらのレポートは、W&B Tables のさまざまなユースケースを強調しています:

* [Visualize Predictions Over Time](https://wandb.ai/stacey/mnist-viz/reports/Visualize-Predictions-over-Time--Vmlldzo1OTQxMTk)
* [How to Compare Tables in Workspaces](https://wandb.ai/stacey/xtable/reports/How-to-Compare-Tables-in-Workspaces--Vmlldzo4MTc0MTA)
* [Image & Classification Models](https://wandb.ai/stacey/mendeleev/reports/Tables-Tutorial-Visualize-Data-for-Image-Classification--VmlldzozNjE3NjA)
* [Text & Generative Language Models](https://wandb.ai/stacey/nlg/reports/Tables-Tutorial-Visualize-Text-Data-Predictions---Vmlldzo1NzcwNzY)
* [Named Entity Recognition](https://wandb.ai/stacey/ner_spacy/reports/Named-Entity-Recognition--Vmlldzo3MDE3NzQ)
* [AlphaFold Proteins](https://wandb.ai/wandb/examples/reports/AlphaFold-ed-Proteins-in-W-B-Tables--Vmlldzo4ODc0MDc)
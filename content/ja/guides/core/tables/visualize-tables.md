---
title: Visualize and analyze tables
description: W&B Tables を可視化し、分析します。
menu:
  default:
    identifier: ja-guides-core-tables-visualize-tables
    parent: tables
weight: 2
---

W&B Tables をカスタマイズして、機械学習モデルのパフォーマンスに関する質問に答えたり、データを分析したりできます。

データをインタラクティブに探索して、以下を実現します。

* モデル、エポック、または個々のサンプル全体で、変更を正確に比較する
* データ内のより高レベルのパターンを理解する
* 視覚的なサンプルで洞察を捉え、伝達する

{{% alert %}}
W&B Tables は、以下の 振る舞 いをします。
1.  **Artifacts コンテキストではステートレス**: Artifacts バージョンとともに記録されたテーブルは、ブラウザウィンドウを閉じるとデフォルトの状態にリセットされます。
2.  **ワークスペース または report コンテキストではステートフル**: 単一の run ワークスペース、マルチ run の project ワークスペース、または Report 内のテーブルに対する変更は保持されます。

現在の W&B Table ビューを保存する方法については、[ビューの保存]({{< relref path="#save-your-view" lang="ja" >}})を参照してください。
{{% /alert %}}

## 2つのテーブルの表示方法
[マージされたビュー]({{< relref path="#merged-view" lang="ja" >}})または[並べて表示するビュー]({{< relref path="#side-by-side-view" lang="ja" >}})で2つのテーブルを比較します。たとえば、下の画像は MNIST データのテーブル比較を示しています。

{{< img src="/images/data_vis/table_comparison.png" alt="左：1回のトレーニングエポック後の間違い、右：5エポック後の間違い" max-width="90%" >}}

次の手順に従って、2つのテーブルを比較します。

1. W&B App で project に移動します。
2. 左側の パネル で Artifacts アイコンを選択します。
3. Artifacts バージョンを選択します。

次の画像では、5つのエポックのそれぞれで MNIST 検証データに対するモデルの予測を示しています ([インタラクティブな例はこちら](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json))。

{{< img src="/images/data_vis/preds_mnist.png" alt="[予測]をクリックしてテーブルを表示します" max-width="90%" >}}

4. サイドバー で比較する 2 番目の Artifacts バージョンにカーソルを合わせ、表示されたら [**比較**] をクリックします。たとえば、下の画像では、5 エポックの トレーニング 後に同じモデルによって行われた MNIST 予測と比較するために、"v4" というラベルの付いた バージョン を選択します。

{{< img src="/images/data_vis/preds_2.png" alt="1 エポック (ここに表示) と 5 エポック (v4) のトレーニング後のモデル予測を比較する準備" max-width="90%" >}}

### マージされたビュー

最初は、両方のテーブルがマージされて表示されます。最初に選択したテーブルにはインデックス0と青色のハイライトが付き、2番目のテーブルにはインデックス1と黄色のハイライトが付きます。[マージされたテーブルのライブ例はこちら](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json#7dd0cd845c0edb469dec)をご覧ください。

{{< img src="/images/data_vis/merged_view.png" alt="マージされたビューでは、数値列はデフォルトでヒストグラムとして表示されます" max-width="90%">}}

マージされたビューから、次のことができます。

* **結合キーを選択する**: 左上のドロップダウンを使用して、2つのテーブルの結合キーとして使用する列を設定します。通常、これはデータセット 内の特定の例のファイル名や、生成されたサンプル のインクリメント インデックスなど、各行の一意の識別子です。現在、_任意の_ 列を選択できるため、判読できないテーブルや低速なクエリが発生する可能性があることに注意してください。
* **結合の代わりに連結する**: このドロップダウンで [すべてのテーブルを連結する] を選択して、列全体を結合する代わりに、両方のテーブルから_すべての行を結合_して1つの大きな Table にします。
* **各 Table を明示的に参照する**: フィルタ式で 0、1、および * を使用して、一方または両方の Table インスタンスの列を明示的に指定します。
* **詳細な数値の差異をヒストグラムとして可視化する**: 一目で任意のセルの値を比較します。

### 並べて表示するビュー

2つのテーブルを並べて表示するには、最初のドロップダウンを [テーブルのマージ：テーブル] から [リスト：テーブル] に変更し、[ページサイズ] をそれぞれ更新します。ここでは、最初に選択した Table が左側に、2番目の Table が右側にあります。また、[垂直] チェックボックスをクリックして、これらのテーブルを垂直方向に比較することもできます。

{{< img src="/images/data_vis/side_by_side.png" alt="並べて表示するビューでは、Table の行は互いに独立しています。" max-width="90%" >}}

* **テーブルを一目で比較する**: 両方のテーブルに (並べ替え、フィルタリング、グループ化) 操作を同時に適用し、変更や差異をすばやく見つけます。たとえば、推測でグループ化された不正な予測、全体で最も難しいネガティブ、真のラベルごとの信頼度スコア分布などを表示します。
* **2つのテーブルを個別に探索する**: スクロールして、目的のサイド/行に焦点を当てます。

## Artifacts を比較する
[経時的にテーブルを比較]({{< relref path="#compare-tables-across-time" lang="ja" >}}) したり、[モデル バリアント を比較]({{< relref path="#compare-tables-across-model-variants" lang="ja" >}}) したりすることもできます。

### 経時的にテーブルを比較する
トレーニング の意味のあるステップごとに Artifacts にテーブルを記録して、トレーニング 時間中のモデルのパフォーマンスを分析します。たとえば、すべての検証ステップの終了時、トレーニング の 50 エポックごと、または パイプライン に適した頻度でテーブルを記録できます。並べて表示するビューを使用して、モデルの予測の変化を可視化します。

{{< img src="/images/data_vis/compare_across_time.png" alt="ラベルごとに、モデルは 1 回のトレーニングエポック (L) よりも 5 回のトレーニングエポック (R) の方が間違いが少なくなります" max-width="90%" >}}

トレーニング 時間中の予測の可視化の詳細なチュートリアルについては、[このレポート](https://wandb.ai/stacey/mnist-viz/reports/Visualize-Predictions-over-Time--Vmlldzo1OTQxMTk)とこのインタラクティブな[ノートブック の例](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/datasets-predictions/W%26B_Tables_Quickstart.ipynb?_gl=1*kf20ui*_gcl_au*OTI3ODM1OTcyLjE3MzE0MzU1NjU.*_ga*ODEyMjQ4MjkyLjE3MzE0MzU1NjU.*_ga_JH1SJHJQXJ*MTczMTcwNTMwNS45LjEuMTczMTcwNTM5My4zMy4wLjA.*_ga_GMYDGNGKDT*MTczMTcwNTMwNS44LjEuMTczMTcwNTM5My4wLjAuMA..)を参照してください。

### モデル バリアント 間でテーブルを比較する

2つの異なるモデルに対して同じステップで記録された2つの Artifacts バージョンを比較して、異なる 設定 (ハイパーパラメーター 、ベース アーキテクチャー など) 全体でモデルのパフォーマンスを分析します。

たとえば、`ベースライン` と新しいモデル バリアント `2x_layers_2x_lr` の間の予測を比較します。ここで、最初の畳み込みレイヤーは 32 から 64 に、2番目の畳み込みレイヤーは 128 から 256 に、学習率は 0.001 から 0.002 に倍増します。[このライブ例](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json#2bb3b1d40aa777496b5d$2x_layers_2x_lr)から、並べて表示するビューを使用し、1 (左側のタブ) 対 5 トレーニング エポック (右側のタブ) 後の不正な予測までフィルタリングします。

{{< tabpane text=true >}}
{{% tab header="1 トレーニングエポック" value="one_epoch" %}}
{{< img src="/images/data_vis/compare_across_variants.png" alt="1 エポック後、パフォーマンスは混在しています。一部のクラスでは精度が向上し、他のクラスでは悪化しています。" >}}
{{% /tab %}}
{{% tab header="5 トレーニングエポック" value="five_epochs" %}}
{{< img src="/images/data_vis/compare_across_variants_after_5_epochs.png" alt="5 エポック後、[ダブル] バリアント は ベースライン に追いついています。" >}}
{{% /tab %}}
{{< /tabpane >}}

## ビューを保存する

run ワークスペース 、 project ワークスペース 、または Report で操作する Tables は、ビューの状態を自動的に保存します。テーブル操作を適用してブラウザを閉じると、次にテーブルに移動したときに、テーブルは最後に表示された 設定 を保持します。

{{% alert %}}
Artifacts コンテキストで操作する Tables はステートレスのままです。
{{% /alert %}}

ワークスペース から特定の状態でテーブルを保存するには、W&B Report にエクスポートします。テーブルを Report にエクスポートするには:
1. ワークスペース 可視化 パネル の右上隅にあるケバブ アイコン (3つの垂直ドット) を選択します。
2. [**パネルを共有**] または [**レポートに追加**] のいずれかを選択します。

{{< img src="/images/data_vis/share_your_view.png" alt="[パネルを共有] を選択すると新しい Report が作成され、[レポートに追加] を選択すると既存の Report に追加できます。" max-width="90%">}}

## 例

これらの Reports は、W&B Tables のさまざまな ユースケース を示しています。

* [経時的な予測の可視化](https://wandb.ai/stacey/mnist-viz/reports/Visualize-Predictions-over-Time--Vmlldzo1OTQxMTk)
* [ワークスペース でテーブルを比較する方法](https://wandb.ai/stacey/xtable/reports/How-to-Compare-Tables-in-Workspaces--Vmlldzo4MTc0MTA)
* [画像と分類モデル](https://wandb.ai/stacey/mendeleev/reports/Tables-Tutorial-Visualize-Data-for-Image-Classification--VmlldzozNjE3NjA)
* [テキストと生成言語モデル](https://wandb.ai/stacey/nlg/reports/Tables-Tutorial-Visualize-Text-Data-Predictions---Vmlldzo1NzcwNzY)
* [固有表現認識](https://wandb.ai/stacey/ner_spacy/reports/Named-Entity-Recognition--Vmlldzo3MDE3NzQ)
* [AlphaFold タンパク質](https://wandb.ai/wandb/examples/reports/AlphaFold-ed-Proteins-in-W-B-Tables--Vmlldzo4ODc0MDc)

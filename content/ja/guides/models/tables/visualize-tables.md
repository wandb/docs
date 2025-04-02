---
title: Visualize and analyze tables
description: W&B Tables を可視化、分析します。
menu:
  default:
    identifier: ja-guides-models-tables-visualize-tables
    parent: tables
weight: 2
---

W&B Tables をカスタマイズして、 機械学習 モデルの性能に関する質問に答えたり、 データを分析したりできます。

インタラクティブにデータを探索して、以下を実現します。

* モデル、 エポック、 個々のサンプル間で、 変更を正確に比較する
* データにおけるより高レベルのパターンを理解する
* 可視化 サンプルでインサイトを捉え、 伝達する

{{% alert %}}
W&B Tables は、 次の 振る舞い をします。
1. **Artifacts コンテキストではステートレス**: Artifacts バージョンとともに記録されたテーブルは、 ブラウザ ウィンドウを閉じるとデフォルト状態にリセットされます。
2. **Workspace または Report コンテキストではステートフル**: 単一 run の Workspace 、 複数 run の Project Workspace 、 または Report でテーブルに加えた変更は保持されます。

現在の W&B Table ビューを保存する方法については、 [ビューを保存]({{< relref path="#save-your-view" lang="ja" >}}) を参照してください。
{{% /alert %}}

## 2つのテーブルを表示する方法
[マージされたビュー]({{< relref path="#merged-view" lang="ja" >}}) または [並べて表示するビュー]({{< relref path="#side-by-side-view" lang="ja" >}}) で2つのテーブルを比較します。 例えば、 下の図はMNIST データのテーブル比較を示しています。

{{< img src="/images/data_vis/table_comparison.png" alt="左: 1 トレーニング エポック後の間違い、 右: 5 エポック後の間違い" max-width="90%" >}}

次の手順に従って、 2つのテーブルを比較します。

1. W&B App で Project に移動します。
2. 左側の パネル で Artifacts アイコンを選択します。
3. Artifacts バージョンを選択します。

次の図では、 5 エポック ([インタラクティブな例はこちら](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json)) の後、 MNIST 検証データに対するモデルの 予測 を示します。

{{< img src="/images/data_vis/preds_mnist.png" alt="[予測] をクリックしてテーブルを表示します" max-width="90%" >}}

4. サイドバー で比較する2番目の Artifacts バージョンにマウスを合わせ、 表示されたら [比較] をクリックします。 例えば、 下の図では、 トレーニング の5 エポック後に同じモデルによって行われたMNIST の 予測 と比較するために、 「v4」 というラベルの バージョン を選択します。

{{< img src="/images/data_vis/preds_2.png" alt="1 エポック (ここに表示) 対 5 エポック (v4) の トレーニング 後にモデルの 予測 を比較する準備" max-width="90%" >}}

### マージされたビュー

最初は、 両方のテーブルがマージされて表示されます。 最初に選択されたテーブルのインデックスは0で青色で強調表示され、 2番目のテーブルのインデックスは1で黄色で強調表示されます。[マージされたテーブルのライブ例はこちら](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json#7dd0cd845c0edb469dec) をご覧ください。

{{< img src="/images/data_vis/merged_view.png" alt="マージされたビューでは、 数値列はデフォルトでヒストグラムとして表示されます" max-width="90%">}}

マージされたビューからは、 以下が可能です。

* **結合キーを選択する**: 左上のドロップダウンを使用して、 2つのテーブルの結合キーとして使用する列を設定します。 通常、 これは データセット 内の特定の例のファイル名や、 生成されたサンプル のインクリメント インデックスなど、 各行の一意の識別子です。 現在、 _任意の_ 列を選択できますが、 判読できないテーブルやクエリの速度低下が発生する可能性があることに注意してください。
* **結合する代わりに連結する**: このドロップダウンで [すべてのテーブルを連結する] を選択すると、 列全体を結合する代わりに、 両方のテーブルから _すべての行を結合_ して、 より大きなテーブルにすることができます。
* **各テーブルを明示的に参照する**: フィルター式で0、 1、 \* を使用して、 1つまたは両方のテーブル インスタンス の列を明示的に指定します。
* **詳細な数値の差をヒストグラムとして可視化する**: 任意のセルの値を一目で比較できます。

### 並べて表示するビュー

2つのテーブルを並べて表示するには、 最初のドロップダウンを [テーブルをマージ: テーブル] から [リスト: テーブル] に変更し、 それぞれ [ページ サイズ] を更新します。 ここで、 最初に選択したテーブルは左側にあり、 2番目のテーブルは右側にあります。 また、 [垂直] チェックボックスをクリックして、 これらのテーブルを垂直方向に比較することもできます。

{{< img src="/images/data_vis/side_by_side.png" alt="並べて表示するビューでは、 テーブルの行は互いに独立しています。" max-width="90%" >}}

* **テーブルを一目で比較する**: 任意の操作 (ソート、 フィルタリング、 グループ化) を両方のテーブルにまとめて適用し、 変更や違いをすばやく見つけます。 例えば、 推測でグループ化された不正な 予測 、 全体で最も難しいネガティブ、 真のラベル ごとの信頼度スコア分布などを表示します。
* **2つのテーブルを個別に探索する**: スクロールして、 目的の側面/行に焦点を当てます。

## Artifacts を比較する
[時間の経過とともにテーブルを比較]({{< relref path="#compare-tables-across-time" lang="ja" >}}) したり、 [モデル バリアント を比較]({{< relref path="#compare-tables-across-model-variants" lang="ja" >}}) したりすることもできます。

### 時間の経過とともにテーブルを比較する
トレーニング の意味のあるステップごとに Artifacts にテーブルを記録して、 トレーニング 時間中のモデルの性能を分析します。 例えば、 すべての検証ステップの終了時、 トレーニング の50 エポックごと、 または パイプライン に適した頻度でテーブルを記録できます。 並べて表示するビューを使用して、 モデル 予測 の変化を可視化します。

{{< img src="/images/data_vis/compare_across_time.png" alt="ラベルごとに、 モデルは1 (L) よりも5 (R) トレーニング エポック後に間違いが少なくなります" max-width="90%" >}}

トレーニング 時間中の 予測 の可視化に関するより詳細なウォークスルーについては、 [この Report](https://wandb.ai/stacey/mnist-viz/reports/Visualize-Predictions-over-Time--Vmlldzo1OTQxMTk) およびこのインタラクティブな [ノートブック の例](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/datasets-predictions/W%26B_Tables_Quickstart.ipynb?_gl=1*kf20ui*_gcl_au*OTI3ODM1OTcyLjE3MzE0MzU1NjU.*_ga*ODEyMjQ4MjkyLjE3MzE0MzU1NjU.*_ga_JH1SJHJQXJ*MTczMTcwNTMwNS45LjEuMTczMTcwNTM5My4zMy4wLjA.*_ga_GMYDGNGKDT*MTczMTcwNTMwNS44LjEuMTczMTcwNTM5My4wLjAuMA..) を参照してください。

### モデル バリアント 全体でテーブルを比較する

2つの異なるモデルに対して同じステップで記録された2つの Artifacts バージョンを比較して、 異なる 設定 (ハイパーパラメーター 、 ベース アーキテクチャー など) 全体でモデルの性能を分析します。

例えば、 `baseline` と新しいモデル バリアント である `2x_layers_2x_lr` の間で 予測 を比較します。この場合、 最初の畳み込みレイヤーは32から64に、 2番目のレイヤーは128から256に、 学習率は0.001から0.002に倍増します。[このライブ例](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json#2bb3b1d40aa777496b5d$2x_layers_2x_lr) から、 並べて表示するビューを使用して、 1 (左側のタブ) 対 5 トレーニング エポック (右側のタブ) 後の不正な 予測 に絞り込みます。

{{< tabpane text=true >}}
{{% tab header="1 トレーニング エポック" value="one_epoch" %}}
{{< img src="/images/data_vis/compare_across_variants.png" alt="1 エポック後、 性能は混合しています: 精度は一部のクラスでは向上し、 他のクラスでは悪化します。" >}}
{{% /tab %}}
{{% tab header="5 トレーニング エポック" value="five_epochs" %}}
{{< img src="/images/data_vis/compare_across_variants_after_5_epochs.png" alt="5 エポック後、 [ダブル] バリアント は ベースライン に追いついています。" >}}
{{% /tab %}}
{{< /tabpane >}}

## ビューを保存

run Workspace 、 Project Workspace 、 または Report で操作するテーブルは、 ビューの状態を自動的に保存します。 テーブル操作を適用して ブラウザ を閉じると、 次にテーブルに移動したときに、 テーブルには最後に表示された 設定 が保持されます。

{{% alert %}}
Artifacts コンテキストで操作するテーブルはステートレスのままです。
{{% /alert %}}

特定の状態で Workspace からテーブルを保存するには、 W&B Report にエクスポートします。 テーブルを Report にエクスポートするには、 次の手順を実行します。
1. Workspace 可視化 パネル の右上隅にあるケバブ アイコン (3つの垂直ドット) を選択します。
2. [パネルを共有] または [レポートに追加] を選択します。

{{< img src="/images/data_vis/share_your_view.png" alt="[パネルを共有] を選択すると新しい Report が作成され、 [レポートに追加] を選択すると既存の Report に追加できます。" max-width="90%">}}

## 例

これらの Reports は、 W&B Tables のさまざまな ユースケース を示しています。

* [時間の経過とともに 予測 を可視化](https://wandb.ai/stacey/mnist-viz/reports/Visualize-Predictions-over-Time--Vmlldzo1OTQxMTk)
* [Workspace でテーブルを比較する方法](https://wandb.ai/stacey/xtable/reports/How-to-Compare-Tables-in-Workspaces--Vmlldzo4MTc0MTA)
* [画像と分類モデル](https://wandb.ai/stacey/mendeleev/reports/Tables-Tutorial-Visualize-Data-for-Image-Classification--VmlldzozNjE3NjA)
* [テキストと生成言語モデル](https://wandb.ai/stacey/nlg/reports/Tables-Tutorial-Visualize-Text-Data-Predictions---Vmlldzo1NzcwNzY)
* [固有表現認識](https://wandb.ai/stacey/ner_spacy/reports/Named-Entity-Recognition--Vmlldzo3MDE3NzQ)
* [AlphaFold タンパク質](https://wandb.ai/wandb/examples/reports/AlphaFold-ed-Proteins-in-W-B-Tables--Vmlldzo4ODc0MDc)

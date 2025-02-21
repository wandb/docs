---
title: Visualize and analyze tables
description: W&B Tables を可視化および分析します。
menu:
  default:
    identifier: ja-guides-core-tables-visualize-tables
    parent: tables
weight: 2
---

W&B Tables をカスタマイズして、機械学習モデルのパフォーマンスについての質問に答えたり、データを分析したりできます。

データをインタラクティブに探索して以下を行います：

* モデル、エポック、または個々の例での変更を正確に比較する
* データの高レベルなパターンを理解する
* ビジュアルサンプルで洞察を捉え、共有する

{{% alert %}}
W&B Tables には以下の振る舞いがあります：
1. **アーティファクトコンテキストでのステートレス**：アーティファクトバージョンと共にログされたテーブルは、ブラウザウィンドウを閉じた後、デフォルトの状態にリセットされます。
2. **ワークスペースまたはレポートコンテキストでのステートフル**：単一のrunワークスペース、複数runプロジェクトワークスペース、またはレポートで行ったテーブルの変更は保持されます。

現在の W&B Table ビューを保存する方法については、[ビューの保存]({{< relref path="#save-your-view" lang="ja" >}})をご覧ください。
{{% /alert %}}

## 2つのテーブルを見る方法
[マージビュー]({{< relref path="#merged-view" lang="ja" >}})または[サイドバイサイドビュー]({{< relref path="#side-by-side-view" lang="ja" >}})を使用して2つのテーブルを比較します。例えば、以下の画像はMNISTデータのテーブル比較を示しています。

{{< img src="/images/data_vis/table_comparison.png" alt="左: 1つのトレーニングエポック後のミス、右: 5エポック後のミス" max-width="90%" >}}

以下のステップで2つのテーブルを比較します：

1. W&B アプリでプロジェクトに移動します。
2. 左のパネルでアーティファクトアイコンを選択します。
3. アーティファクトバージョンを選択します。

以下の画像では、5エポック後のMNIST検証データにおけるモデルの予測を示しています（[ここでインタラクティブな例を表示](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json)）。

{{< img src="/images/data_vis/preds_mnist.png" alt="Tableを見るには 'predictions' をクリックしてください" max-width="90%" >}}

4. 比較したい2番目のアーティファクトバージョンをサイドバーでホバーし、**Compare** が表示されたらクリックします。例えば、以下の画像では "v4" とラベル付けされたバージョンを選択し、同じモデルによる5エポックトレーニング後のMNIST予測と比較します。

{{< img src="/images/data_vis/preds_2.png" alt="1エポック (v0) トレーニング後のモデル予測 (ここに表示) と5エポック (v4) トレーニング後のモデル予測を比較する準備をしています" max-width="90%" >}}

### マージビュ

最初は両方のテーブルがマージされた状態で表示されます。最初に選択されたテーブルはインデックス0と青いハイライトを持ち、2番目のテーブルはインデックス1と黄色のハイライトを持ちます。[ここでマージされたテーブルのライブ例を見る](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json#7dd0cd845c0edb469dec)。

{{< img src="/images/data_vis/merged_view.png" alt="マージビューでは、数値の列がデフォルトでヒストグラムとして表示されます" max-width="90%">}}

マージビューからは、以下を行うことができます

* **結合キーを選択する**：左上にあるドロップダウンを使用して、2つのテーブルを結合する列を設定します。通常、これはデータセット内の特定の例のファイル名や生成されたサンプルの増加インデックスのような、各行の一意の識別子です。現在、任意の列を選択することも可能ですが、これにより読みにくいテーブルや遅いクエリが生成されることがあります。
* **結合ではなく連結する**：このドロップダウンで「すべてのテーブルを結合する」を選択して、両方のテーブルのすべての行を1つの大きなテーブルに結合することができます。
* **各テーブルを明示的に参照する**：0, 1, および \* をフィルター式で使用して、1つまたは両方のテーブルインスタンスで特定の列を明示的に指定します。
* **詳細な数値の違いをヒストグラムとして可視化する**：一目で任意のセルの値を比較します。

### サイドバイサイドビュー

2つのテーブルを並べて表示するには、最初のドロップダウンを「マージテーブル: Table」から「リストに: Table」に変更し、「ページサイズ」をそれに応じて更新します。ここでは、左側に最初に選択したテーブルがあり、右側に2つ目のテーブルがあります。また、「垂直」チェックボックスをクリックすることでこれらのテーブルを縦に比較することもできます。

{{< img src="/images/data_vis/side_by_side.png" alt="サイドバイサイドビューでは、テーブル行は互いに独立しています。" max-width="90%" >}}

* **テーブルを一目で比較する**：あらゆる操作（ソート、フィルター、グループ）を両方のテーブルに適用し、すばやく変更や違いを確認します。例えば、間違った予測を推測によってグループ化し全体で最も難しいネガティブを表示したり、実際のラベルによって信頼度スコアの分布を表示したりします。
* **2つのテーブルを独立して探索する**：スクロールして関心のあるサイド/行に焦点を当てます。

## アーティファクトを比較する
[時間の経過と共にテーブルを比較する]({{< relref path="#compare-tables-across-time" lang="ja" >}})または[モデルのバリエーションを比較する]({{< relref path="#compare-tables-across-model-variants" lang="ja" >}})こともできます。

### 時間の経過と共にテーブルを比較する
モデルパフォーマンスを分析するために、トレーニングの各意味のあるステップでアーティファクトにテーブルをログします。例えば、各検証ステップの終わりに、トレーニングの50エポックごとに、またはあなたのパイプラインにとって意味のある頻度でテーブルをログします。サイドバイサイドビューを使用してモデルの予測の変化を可視化します。

{{< img src="/images/data_vis/compare_across_time.png" alt="各ラベルについて、モデルは1トレーニングエポック (L)後よりも5エポック (R)後にはより少ないミスを犯します" max-width="90%" >}}

トレーニング時間にわたる予測を可視化するより詳細な手順については、[このレポートをご覧ください](https://wandb.ai/stacey/mnist-viz/reports/Visualize-Predictions-over-Time--Vmlldzo1OTQxMTk)そしてこのインタラクティブな[ノートブック例](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/datasets-predictions/W%26B_Tables_Quickstart.ipynb?_gl=1*kf20ui*_gcl_au*OTI3ODM1OTcyLjE3MzE0MzU1NjU.*_ga*ODEyMjQ4MjkyLjE3MzE0MzU1NjU.*_ga_JH1SJHJQXJ*MTczMTcwNTMwNS45LjEuMTczMTcwNTM5My4zMy4wLjA.*_ga_GMYDGNGKDT*MTczMTcwNTMwNS44LjEuMTczMTcwNTM5My4wLjAuMA..)をご覧ください。

### モデルのバリエーションを比較する
異なるモデルで同じステップでログされた2つのアーティファクトバージョンを比較して、異なる設定（ハイパーパラメーター、基本アーキテクチャーなど）におけるモデルのパフォーマンスを分析します。

例えば、`baseline` と新しいモデルバリアント `2x_layers_2x_lr` の間で予測を比較します。ここで最初の畳み込み層は32から64に倍増し、2番目は128から256に倍増し、学習率は0.001から0.002に倍増します。[このライブ例](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json#2bb3b1d40aa777496b5d$2x_layers_2x_lr)から、サイドバイサイドビューを使用して、1トレーニングエポック（左タブ）と5トレーニングエポック（右タブ）後の間違った予測に絞り込みます。

{{< tabpane text=true >}}
{{% tab header="1 training epoch" value="one_epoch" %}}
{{< img src="/images/data_vis/compare_across_variants.png" alt="1エポック後、パフォーマンスは混合しており: 一部のクラスでは精度が向上し、他のクラスでは悪化します。" >}}
{{% /tab %}}
{{% tab header="5 training epochs" value="five_epochs" %}}
{{< img src="/images/data_vis/compare_across_variants_after_5_epochs.png" alt="5エポック後、「ダブル」バリアントはベースラインに追いついています。" >}}
{{% /tab %}}
{{< /tabpane >}}

## ビューを保存する

run ワークスペース、プロジェクトワークスペース、またはレポートで操作したテーブルは、自動的にビューの状態を保存します。テーブル操作を適用してブラウザを閉じた場合、次にそのテーブルに移動したときに、最後に表示した設定が保持されます。

{{% alert %}}
アーティファクトコンテキストで操作したテーブルはステートレスのままです。
{{% /alert %}}

特定の状態でワークスペースからテーブルを保存するには、W&Bレポートにエクスポートします。レポートにテーブルをエクスポートするには：
1. ワークスペースの可視化パネルの右上隅にあるケバブアイコン（三つの縦のドット）を選択します。
2. **Share panel** または **Add to report** のいずれかを選択します。

{{< img src="/images/data_vis/share_your_view.png" alt="Share panel 新しいレポートを作成、Add to report で既存のレポートに追加します。" max-width="90%">}}

## 例

これらのレポートは、W&B Tables のさまざまなユースケースを強調しています：

* [時間の経過による予測の可視化](https://wandb.ai/stacey/mnist-viz/reports/Visualize-Predictions-over-Time--Vmlldzo1OTQxMTk)
* [ワークスペースでテーブルを比較する方法](https://wandb.ai/stacey/xtable/reports/How-to-Compare-Tables-in-Workspaces--Vmlldzo4MTc0MTA)
* [画像と分類モデル](https://wandb.ai/stacey/mendeleev/reports/Tables-Tutorial-Visualize-Data-for-Image-Classification--VmlldzozNjE3NjA)
* [テキストと生成言語モデル](https://wandb.ai/stacey/nlg/reports/Tables-Tutorial-Visualize-Text-Data-Predictions---Vmlldzo1NzcwNzY)
* [固有表現抽出](https://wandb.ai/stacey/ner_spacy/reports/Named-Entity-Recognition--Vmlldzo3MDE3NzQ)
* [AlphaFold タンパク質](https://wandb.ai/wandb/examples/reports/AlphaFold-ed-Proteins-in-W-B-Tables--Vmlldzo4ODc0MDc)
---
description: W&B Tablesを視覚化および分析する。
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# テーブルの可視化と分析

W&B Tablesをカスタマイズして、機械学習モデルのパフォーマンスに関する質問に答えたり、データを分析したりできます。

データをインタラクティブに探索して以下を行います：

* モデル、エポック、または個々の例にわたる変化を正確に比較
* データ内の高次パターンを理解
* 視覚的なサンプルで洞察をキャプチャし、伝達

:::info
W&B Tablesは以下の振る舞いを持ちます：
1. **アーティファクトコンテキストでのステートレス**：アーティファクトバージョンと一緒にログされたテーブルは、ブラウザウィンドウを閉じるとデフォルトの状態にリセットされます。
2. **ワークスペースまたはレポートコンテキストでのステートフル**：単一のrunワークスペース、複数runプロジェクトワークスペース、またはレポート内でテーブルに加えた変更は持続します。

現在のW&B Tableビューを保存する方法については、[ビューの保存](#save-your-view)を参照してください。
:::

## 2つのテーブルを表示する方法
[マージビュー](#merged-view)または[サイドバイサイドビュー](#side-by-side-view)で2つのテーブルを比較します。例えば、以下の画像はMNISTデータのテーブル比較を示しています。

![左：1エポック後のミス、右：5エポック後のミス](/images/data_vis/table_comparison.png)

以下の手順に従って2つのテーブルを比較します：

1. W&Bアプリでプロジェクトに移動します。
2. 左側のパネルでアーティファクトアイコンを選択します。
3. アーティファクトバージョンを選択します。

以下の画像では、5エポック後のMNIST検証データに対するモデルの予測を示しています（[インタラクティブな例はこちら](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json)）。

![「predictions」をクリックしてテーブルを表示](@site/static/images/data_vis/preds_mnist.png)

4. サイドバーで比較したい2つ目のアーティファクトバージョンにカーソルを合わせ、表示されたら**Compare**をクリックします。例えば、以下の画像では、1エポック後の予測（v0）と5エポック後の予測（v4）を比較するために「v4」とラベル付けされたバージョンを選択します。

![1エポック後のトレーニング（v0、ここに表示）と5エポック後のトレーニング（v4）のモデル予測を比較する準備](@site/static/images/data_vis/preds_2.png)

### マージビュー

最初に、両方のテーブルがマージされた状態で表示されます。最初に選択したテーブルはインデックス0で青色のハイライトが付き、2つ目のテーブルはインデックス1で黄色のハイライトが付きます。[マージされたテーブルのライブ例はこちら](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json#7dd0cd845c0edb469dec)。

![マージビューでは、数値列はデフォルトでヒストグラムとして表示されます](@site/static/images/data_vis/merged_view.png)

マージビューから以下の操作が可能です：

* **結合キーの選択**：左上のドロップダウンを使用して、2つのテーブルを結合するための列を設定します。通常、これは各行の一意の識別子（例：データセット内の特定の例のファイル名や生成されたサンプルのインクリメントインデックス）になります。現在、_任意の_列を選択することが可能ですが、読みづらいテーブルやクエリの遅延を引き起こす可能性があります。
* **結合の代わりに連結**：このドロップダウンで「すべてのテーブルを連結」を選択して、両方のテーブルのすべての行を1つの大きなテーブルに_結合_する代わりに、列をまたいで結合します。
* **各テーブルを明示的に参照**：フィルター式で0、1、および\*を使用して、1つまたは両方のテーブルインスタンスの列を明示的に指定します。
* **詳細な数値差をヒストグラムとして可視化**：任意のセルの値を一目で比較します。

### サイドバイサイドビュー

2つのテーブルをサイドバイサイドで表示するには、最初のドロップダウンを「Merge Tables: Table」から「List of: Table」に変更し、「Page size」をそれぞれ更新します。ここでは、最初に選択したテーブルが左側に、2つ目のテーブルが右側に表示されます。また、「Vertical」チェックボックスをクリックして、これらのテーブルを縦に比較することもできます。

![サイドバイサイドビューでは、テーブルの行は互いに独立しています。](/images/data_vis/side_by_side.png)

* **テーブルを一目で比較**：任意の操作（ソート、フィルター、グループ）を両方のテーブルに同時に適用し、変更や違いをすばやく確認します。例えば、予測の誤りを推測別に表示したり、全体で最も難しいネガティブを表示したり、真のラベルごとの信頼スコア分布を表示したりします。
* **2つのテーブルを独立して探索**：興味のあるサイドや行をスクロールしてフォーカスします。

## アーティファクトの比較
[時間の経過にわたるテーブルの比較](#compare-across-time)や[モデルバリアントの比較](#compare-across-model-variants)も可能です。

### 時間の経過にわたるテーブルの比較
トレーニングの各重要なステップでアーティファクトにテーブルをログして、トレーニング時間にわたるモデルのパフォーマンスを分析します。例えば、各検証ステップの終わりに、50エポックごとに、または開発フローに適した任意の頻度でテーブルをログできます。サイドバイサイドビューを使用して、モデル予測の変化を可視化します。

![各ラベルに対して、モデルは5エポック後（右）に1エポック後（左）よりも少ないミスをします。](/images/data_vis/compare_across_time.png)

トレーニング時間にわたる予測の可視化に関する詳細なウォークスルーについては、[このレポート](https://wandb.ai/stacey/mnist-viz/reports/Visualize-Predictions-over-Time--Vmlldzo1OTQxMTk)およびこのインタラクティブな[ノートブック例](http://wandb.me/tables-walkthrough)を参照してください。

### モデルバリアントの比較

異なるモデルの2つのアーティファクトバージョンを同じステップでログして、異なる設定（ハイパーパラメーター、ベースアーキテクチャーなど）にわたるモデルのパフォーマンスを分析します。

例えば、`baseline`と新しいモデルバリアント`2x_layers_2x_lr`の予測を比較します。ここでは、最初の畳み込み層が32から64に、2番目が128から256に、学習率が0.001から0.002に倍増します。[このライブ例](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json#2bb3b1d40aa777496b5d$2x\_layers\_2x\_lr)から、サイドバイサイドビューを使用して、1エポック後（左タブ）と5エポック後（右タブ）の誤った予測にフィルターをかけます。

<Tabs
  defaultValue="one_epoch"
  values={[
    {label: '1 training epoch', value: 'one_epoch'},
    {label: '5 training epochs', value: 'five_epochs'},
  ]}>
  <TabItem value="one_epoch">

![1エポック後、パフォーマンスはクラスによって改善したり悪化したりします。](/images/data_vis/compare_across_variants.png)
  </TabItem>
  <TabItem value="five_epochs">

![5エポック後、「ダブル」バリアントはベースラインに追いついています。](/images/data_vis/compare_across_variants_after_5_epochs.png)
  </TabItem>
</Tabs>

## ビューの保存

runワークスペース、プロジェクトワークスペース、またはレポートで操作したテーブルは、自動的にビュー状態を保存します。テーブル操作を適用してブラウザを閉じた場合、次回テーブルに移動したときに最後に表示した設定が保持されます。

:::tip
アーティファクトコンテキストで操作したテーブルはステートレスのままです。
:::

特定の状態でワークスペースからテーブルを保存するには、W&Bレポートにエクスポートします。レポートにテーブルをエクスポートするには：
1. ワークスペースの可視化パネルの右上隅にあるケバブアイコン（三つの縦の点）を選択します。
2. **Share panel**または**Add to report**を選択します。

![Share panelは新しいレポートを作成し、Add to reportは既存のレポートに追加します。](/images/data_vis/share_your_view.png)

## 例

これらのレポートは、W&B Tablesのさまざまなユースケースを強調しています：

* [Visualize Predictions Over Time](https://wandb.ai/stacey/mnist-viz/reports/Visualize-Predictions-over-Time--Vmlldzo1OTQxMTk)
* [How to Compare Tables in Workspaces](https://wandb.ai/stacey/xtable/reports/How-to-Compare-Tables-in-Workspaces--Vmlldzo4MTc0MTA)
* [Image & Classification Models](https://wandb.ai/stacey/mendeleev/reports/Tables-Tutorial-Visualize-Data-for-Image-Classification--VmlldzozNjE3NjA)
* [Text & Generative Language Models](https://wandb.ai/stacey/nlg/reports/Tables-Tutorial-Visualize-Text-Data-Predictions---Vmlldzo1NzcwNzY)
* [Named Entity Recognition](https://wandb.ai/stacey/ner_spacy/reports/Named-Entity-Recognition--Vmlldzo3MDE3NzQ)
* [AlphaFold Proteins](https://wandb.ai/wandb/examples/reports/AlphaFold-ed-Proteins-in-W-B-Tables--Vmlldzo4ODc0MDc)
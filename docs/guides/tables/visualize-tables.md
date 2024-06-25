---
description: W&B Tables を視覚化および分析する。
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# テーブルの可視化と分析

W&B テーブルをカスタマイズして、機械学習モデルの性能についての質問に答えたり、データを分析したりできます。

インタラクティブにデータを探索して以下のことができます：

* モデル、エポック、または個々のサンプル間の変化を正確に比較
* データの高レベルなパターンを理解
* 視覚的なサンプルを用いて見識を記録・伝達

:::info
W&B テーブルは以下の振る舞いを持ちます：
1. **アーティファクトコンテキストでステートレス**： アーティファクトバージョンと一緒にログされたテーブルは、ブラウザウィンドウを閉じるとデフォルト状態にリセットされます。
2. **ワークスペースまたはレポートコンテキストでステートフル**： 単一のrunワークスペース、複数runプロジェクトワークスペース、またはレポート内でテーブルに加えた変更は維持されます。

現在の W&B テーブルビューを保存する方法については、[ビューを保存](#save-your-view)をご覧ください。
:::

## 2つのテーブルを表示する方法
[結合ビュー](#merged-view) または [並列ビュー](#side-by-side-view)を使用して2つのテーブルを比較します。以下の画像は、MNISTデータのテーブル比較を示しています。

![左: 1エポック後の間違い、右: 5エポック後の間違い](/images/data_vis/table_comparison.png)

以下の手順で2つのテーブルを比較します：

1. W&Bアプリでプロジェクトに移動します。
2. 左のパネルのArtifactsアイコンを選択します。
2. アーティファクトバージョンを選択します。

以下の画像では、5エポック後のMNIST検証データに対するモデルの予測を示しています（[インタラクティブな例を表示](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json)）。

![「予測」をクリックしてテーブルを表示](@site/static/images/data_vis/preds_mnist.png)

3. 比較したい2番目のアーティファクトバージョンにマウスを乗せ、表示される**比較**をクリックします。例えば、以下の画像では、5エポックのトレーニング後のMNIST予測と比較するために「v4」とラベル付けされたバージョンを選択します。

![1エポック後のトレーニング予測モデル（ここではv0）と5エポック後の比較の準備](@site/static/images/data_vis/preds_2.png)

### 結合ビュー

最初に両方のテーブルが結合された状態で表示されます。最初に選択したテーブルにはインデックス0が付いて青くハイライトされ、2番目のテーブルにはインデックス1が付いて黄色にハイライトされます。 [結合テーブルのライブ例をこちらから表示](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json#7dd0cd845c0edb469dec)。

![結合ビューでは、数値列はデフォルトでヒストグラムとして表示されます](@site/static/images/data_vis/merged_view.png)

結合ビューから以下の操作が可能です：

* **結合キーを選択**： 左上のドロップダウンを使用して、2つのテーブルで結合に使用する列を設定します。通常、これは各行の一意の識別子（例えば、データセット内の特定のサンプルのファイル名や生成されたサンプルのインクリメントインデックス）になります。現在、_任意の_ 列を選択することができるため、読みやすくないテーブルやクエリが遅くなる可能性があります。
* **結合ではなく連結**： このドロップダウンで「すべてのテーブルを連結」を選択して、2つのテーブルの行を一つの大きなテーブルに_ユニオン_することができます。
* **各テーブルを明示的に参照**： フィルタ式で0, 1, \*を使用して、1つまたは両方のテーブルインスタンスの列を明示的に指定できます。
* **詳細な数値の違いをヒストグラムで視覚化**： 任意のセルの値を一目で比較できます。

### 並列ビュー

2つのテーブルを並べて表示するには、最初のドロップダウンを「結合テーブル: Table」から「リスト: Table」に変更し、それぞれの「ページサイズ」を更新します。最初に選択したテーブルは左に、2番目のものは右に表示されます。また、「縦方向」チェックボックスをクリックして、テーブルを縦方向に比較することもできます。

![並列ビューでは、各テーブルの行が独立しています。](/images/data_vis/side_by_side.png)

* **一目でテーブルを比較**： 両方のテーブルに一括で操作（並べ替え、フィルタ、グループ）を適用し、変化や違いを迅速に確認できます。例えば、間違った予測をグループ化して表示したり、最も難しい否定例全体、真のラベルによる信頼スコアの分布などを確認できます。
* **2つのテーブルを独立して探索**： 関心のある側や行をスクロールしてフォーカスできます。

## アーティファクトを比較
[時間経過で表を比較](#compare-across-time)または[モデルバリアントを比較](#compare-across-model-variants)することもできます。

### 時間経過で表を比較
トレーニングの各重要なステップでアーティファクトにテーブルをログして、トレーニング時間のモデル性能を分析します。例えば、各検証ステップの終了時にテーブルをログしたり、トレーニングの50エポックごとにログしたり、パイプラインに適した頻度でログすることができます。並列ビューを使用して予測の変化を視覚化します。

![各ラベルに対して、5エポック後（R）のモデルは1エポック後（L）よりもミスが少ない](/images/data_vis/compare_across_time.png)

トレーニング時間の予測を視覚化する詳細な手順については、[こちらのレポート](https://wandb.ai/stacey/mnist-viz/reports/Visualize-Predictions-over-Time--Vmlldzo1OTQxMTk)およびこのインタラクティブな[ノートブックの例](http://wandb.me/tables-walkthrough)をご覧ください。

### モデルバリアントを比較

トレーニングの同一ステップで記録された2つのアーティファクトバージョンを比較して、異なる設定（ハイパーパラメーター、ベースアーキテクチャーなど）でモデルの性能を分析します。

例えば、`baseline`と新しいモデルバリアント`2x_layers_2x_lr`の予測を比較します。このバリアントでは、最初の畳み込み層が32から64に、2番目が128から256に、学習率が0.001から0.002に倍増します。[こちらのライブ例](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json#2bb3b1d40aa777496b5d$2x_layers_2x_lr)から、並列ビューを使用して1エポック後（左タブ）と5エポック後（右タブ）の間違った予測に絞り込みます。

<Tabs
  defaultValue="one_epoch"
  values={[
    {label: '1 training epoch', value: 'one_epoch'},
    {label: '5 training epochs', value: 'five_epochs'},
  ]}>
  <TabItem value="one_epoch">

![1エポック後、クラスによって精度が向上したり悪化したりする。](/images/data_vis/compare_across_variants.png)
  </TabItem>
  <TabItem value="five_epochs">

![5エポック後、「ダブル」バリアントがベースラインに追いついてきている。](/images/data_vis/compare_across_variants_after_5_epochs.png)
  </TabItem>
</Tabs>

## ビューを保存する

runワークスペース、プロジェクトワークスペース、またはレポート内で操作するテーブルは、自動的にそのビュー状態を保存します。テーブル操作を適用した後ブラウザを閉じた場合でも、次回にそのテーブルに移動したときに最後に表示した設定が維持されます。

:::tip
アーティファクトコンテキストで操作するテーブルはステートレスのままです。
:::

特定の状態のテーブルをワークスペースから保存するには、それをW&Bレポートにエクスポートします。レポートにテーブルをエクスポートするには：
1. ワークスペースの可視化パネルの右上隅にあるケボブアイコン（三つの縦の点）を選択します。
2. **パネルの共有**または**レポートに追加**を選択します。

![パネルの共有は新しいレポートを作成し、レポートに追加は既存のレポートに追加します。](/images/data_vis/share_your_view.png)

## 例

これらのレポートは、W&Bテーブルの異なるユースケースをハイライトしています：

* [Visualize Predictions Over Time](https://wandb.ai/stacey/mnist-viz/reports/Visualize-Predictions-over-Time--Vmlldzo1OTQxMTk)
* [How to Compare Tables in Workspaces](https://wandb.ai/stacey/xtable/reports/How-to-Compare-Tables-in-Workspaces--Vmlldzo4MTc0MTA)
* [Image & Classification Models](https://wandb.ai/stacey/mendeleev/reports/Tables-Tutorial-Visualize-Data-for-Image-Classification--VmlldzozNjE3NjA)
* [Text & Generative Language Models](https://wandb.ai/stacey/nlg/reports/Tables-Tutorial-Visualize-Text-Data-Predictions---Vmlldzo1NzcwNzY)
* [Named Entity Recognition](https://wandb.ai/stacey/ner_spacy/reports/Named-Entity-Recognition--Vmlldzo3MDE3NzQ)
* [AlphaFold Proteins](https://wandb.ai/wandb/examples/reports/AlphaFold-ed-Proteins-in-W-B-Tables--Vmlldzo4ODc0MDc)
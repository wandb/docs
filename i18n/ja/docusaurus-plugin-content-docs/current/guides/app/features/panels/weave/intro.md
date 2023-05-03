---

slug: /guides/app/features/panels/weave
description: 
  このページの一部の機能はベータ版で、機能フラグの後ろに隠れています。プロフィールページの自己紹介欄に
  `weave-plot` を追加することで、関連するすべての機能をアンロックできます。
---

# Weave

## はじめに

Weaveパネルは、ユーザーがW&Bに直接データを問い合わせ、結果を可視化し、さらにインタラクティブに分析できるようにします。Weaveパネルには、以下の画像に示すように、主に4つのコンポーネントがあります。

1. **Weave式**: W&Bのバックエンドに対して実行するクエリを指定します
2. **Weaveパネルセレクター**: クエリの結果を表示するために使用されるパネルを指定します。
3. **Weave構成**: ユーザーがWeave式および/またはWeaveパネルのパラメーターを設定できるようにします
4. **Weave結果パネル**: Weaveパネルの主要なエリアで、指定されたWeaveパネルと構成を使用して、Weave式クエリの結果を表示します。

![](/images/weave/weave_panel_components.png)

Weave、テーブル、プロットをすぐに試してみたい場合は、こちらの[インタラクティブレポート](https://wandb.ai/timssweeney/keras\_learning\_rate/reports/Announcing-W-B-Weave-Plot--VmlldzoxMDIyODM1)をご覧ください。

## コンポーネント

### Weave式

Weave式を使用すると、ユーザーはW&Bに格納されたデータ（ラン、アーティファクト、モデル、テーブルなど）を問い合わせることができます！最も一般的なWeave式は、テーブルをログに記録することで生成されます。`wandb.log({"predictions":<MY_TABLE>})`、そして次のようになります。

![](/images/weave/basic_weave_expression.png)

これを分解してみましょう：
* `runs`は、Weaveパネルがワークスペース内にある場合、Weaveパネル式に自動的に挿入される**変数**です。その「値」は、その特定のワークスペースで表示されるランのリストです。[こちらのリンクでrun内で利用できる様々な属性について読む](../../../../track/public-api-guide.md#understanding-the-different-attributes)。
* `summary`は、RunのSummaryオブジェクトを返す**op**です。注意: **ops**は「マップ」されており、この**op**はリスト内の各Runに適用され、Summaryオブジェクトのリストが生成されます。
* `["predictions"]`は、Pick **op**（角括弧で示される）で、**パラメーター**は"predictions"です。Summaryオブジェクトは辞書やマップのように機能するため、この操作では、各Summaryオブジェクトから"predictions"フィールドが「選択」されます。上述のように、「predictions」フィールドはTableであり、このクエリによって上記のTableが生成されます。

Weave式は非常に強力であり、例えば、次の式では:

* 自分のランを`name = "easy-bird-1"`のものだけにフィルタリング
* Summaryオブジェクトを取得
* "Predictions"の値を選択
* テーブルをマージ
* テーブルをクエリ
* 結果をプロット

ここで、マージ、クエリ、およびプロット構成はWeave構成（下記で説明）で指定されています。Ops、Types、およびこのクエリ言語のその他の特徴に関する完全な説明については、Weave Expression Docsを参照してください。

![](/images/weave/merge_query_plot_example.png)

### Weaveパネルセレクタ

Weave式を構築した後、Weaveパネルは自動的に結果を表示するために使用するパネルを選択します。結果のデータタイプに対して最も一般的なパネルが自動的に選択されます。ただし、パネルを変更する場合は、ドロップダウンをクリックして別のパネルを選択します。

![](/images/weave/panel_selector.png)

いくつかの特別なケースに注意してください:

1. 現在テーブルを表示している場合、通常のオプションに加えて、「Plot table query」オプションが利用可能になります。このオプションを選択すると、_現在のテーブルクエリ_の結果をプロットすることを意味します。つまり、カスタムフィールドの追加、グループ化、ソート、フィルタリングなど、テーブルを操作している場合は、`Plot table query`を選択して、現在の結果をプロットの入力として使用できます。
2.  `Merge Tables: <Panel>`は、入力データタイプがテーブルのリストである特別なケースです。このような場合、「Merge Tables」のパネル部分では、すべての行を連結するか、特定の列でテーブルを結合することができます。この設定は、Weave構成（下記で説明）で設定され、次のスクリーンショットに示されています。

    ![](/images/weave/merge_tables_concate.png) ![](/images/weave/merge_tables_join.png)
3.  `List of: <Panel>`は、入力データタイプがリストである特別なケースであり、ページ化されたパネル表示を表示したい場合です。次の例では、`List of: Plot`が表示され、それぞれのプロットが異なるランから来ています。
![](/images/weave/list_of_panels_plot.png)

### Weave設定

パネルの左上隅にある歯車アイコンをクリックして、Weave設定を展開します。これにより、ユーザーは特定の式オペレーションのパラメーターと結果パネルを構成することができます。例えば：

![](/images/weave/config_box_plot.png)result_

上記の例では、展開されたWeave設定に3つのセクションが表示されます。

1. `Merge Tables`: 式内の`merge`オペレーションには、追加の設定プロパティ（この場合はConcatenateまたはJoin）が存在し、ここで公開されます。
2. `Table Query`: 式内の`table`オペレーションは結果に適用されるテーブルクエリを表し、ユーザーは`Edit table query`ボタンをクリックすることでテーブルクエリをインタラクティブに編集できます。
3. `Plot`: 最後に、式オペレーションが設定された後、Result Panel自体が構成可能になります。この場合、`Plot`パネルには、次元とその他のプロット特性を設定するための構成があります。ここでは、x軸にカテゴリカルな正解値、y軸にモデルの"1"クラスの予測スコアを持つボックスプロットが設定されています。期待通り、"1"のスコア分布は他のクラスよりも著しく高いです。

### Weave結果パネル

最後に、Weave結果パネルは、選択されたWeaveパネルを使用して、Weave式の結果をインタラクティブな形式で表示します。ここでは、同じデータのテーブルとプロットが表示されています。

:::info
すべての列を一度に同じサイズにリサイズするには、`shift` + マウスドラッグを使用してリサイズできます。
:::

![](/images/weave/result_panel.png)

![](/images/weave/result_panel_merge_table_plot.png)

## Weaveパネルの作成

ユーザーが[ログにテーブルを記録する](../../../../track/log/log-tables.md)場合や[カスタムチャートをログに記録する](../../custom-charts/intro.md)場合には、Weaveパネルが自動的に作成されます。このような場合、Weave式を`run.summary["<TABLE_NAME>"]`に自動的に設定し、テーブルパネルをレンダリングします。さらに、「パネルの追加」ボタンから`Weave`パネルを選択して、ワークスペースに直接Weaveパネルを追加することができます。
![](/images/weave/create_weave_panel.png)
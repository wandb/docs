---
slug: /guides/app/features/custom-charts
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Custom Charts

**Custom Charts** を使用して、現在のデフォルトUIでは実現できないチャートを作成しましょう。任意のテーブルデータをログし、あなたの望むとおりに可視化できます。フォント、色、ツールチップの詳細を [Vega](https://vega.github.io/vega/) のパワーを使って制御できます。

* **可能なこと**: [launch announcement →](https://wandb.ai/wandb/posts/reports/Announcing-the-W-B-Machine-Learning-Visualization-IDE--VmlldzoyNjk3Nzg) をご覧ください
* **コード**: [ホストされたノートブック →](https://tiny.cc/custom-charts) でライブ例を試してください
* **ビデオ**: [ウォークスルービデオ →](https://www.youtube.com/watch?v=3-N9OV6bkSM) をご覧ください
* **例**: クイック Keras と Sklearn の[デモノートブック →](https://colab.research.google.com/drive/1g-gNGokPWM2Qbc8p1Gofud0\_5AoZdoSD?usp=sharing)

![Supported charts from vega.github.io/vega](/images/app_ui/supported_charts.png)

### 仕組み

1. **データをログする**: あなたのスクリプトから、通常W&Bを使用して実行する際と同じように [config](../../../../guides/track/config.md) データとサマリーデータをログします。特定の時点でログされた複数の値のリストを可視化するには、カスタム `wandb.Table` を使用します。
2. **チャートをカスタマイズする**: このログデータを [GraphQL](https://graphql.org) クエリで取り込みます。クエリの結果を [Vega](https://vega.github.io/vega/) で可視化します。Vegaは強力な可視化文法です。
3. **チャートをログする**: スクリプトから `wandb.plot_table()` を呼び出して、プリセットを使用します。

![](/images/app_ui/pr_roc.png)

## スクリプトからチャートをログする

### 組み込みプリセット

これらのプリセットには組み込みの `wandb.plot` メソッドがあり、スクリプトから直接チャートをログし、探している正確なビジュアライゼーションをUIで素早く確認できます。

<Tabs
  defaultValue="line-plot"
  values={[
    {label: 'Line plot', value: 'line-plot'},
    {label: 'Scatter plot', value: 'scatter-plot'},
    {label: 'Bar chart', value: 'bar-chart'},
    {label: 'Histogram', value: 'histogram'},
    {label: 'PR curve', value: 'pr-curve'},
    {label: 'ROC curve', value: 'roc-curve'},
  ]}>
  <TabItem value="line-plot">

`wandb.plot.line()`

カスタムラインプロットをログします—任意の軸xとy上の接続された順序付きポイント(x,y)のリスト。

```python
data = [[x, y] for (x, y) in zip(x_values, y_values)]
table = wandb.Table(data=data, columns=["x", "y"])
wandb.log(
    {
        "my_custom_plot_id": wandb.plot.line(
            table, "x", "y", title="Custom Y vs X Line Plot"
        )
    }
)
```

任意の2次元のカーブをログするために使用できます。リストの値をプロットする場合、リストの値の数が正確に一致する必要があります（つまり、各ポイントにはxとyが必要です）。

![](/images/app_ui/line_plot.png)

[アプリ内で確認 →](https://wandb.ai/wandb/plots/reports/Custom-Line-Plots--VmlldzoyNjk5NTA)

[コードを実行 →](https://tiny.cc/custom-charts)

  </TabItem>
  <TabItem value="scatter-plot">

`wandb.plot.scatter()`

カスタム散布図をログします—任意の軸xとy上のポイント(x, y)のリスト。

```python
data = [[x, y] for (x, y) in zip(class_x_prediction_scores, class_y_prediction_scores)]
table = wandb.Table(data=data, columns=["class_x", "class_y"])
wandb.log({"my_custom_id": wandb.plot.scatter(table, "class_x", "class_y")})
```

任意の二次元の散布ポイントをログするために使用できます。リストの値をプロットする場合、リストの値の数が正確に一致する必要があります（つまり、各ポイントにはxとyが必要です）。

![](/images/app_ui/demo_scatter_plot.png)

[アプリ内で確認 →](https://wandb.ai/wandb/plots/reports/Custom-Scatter-Plots--VmlldzoyNjk5NDQ)

[コードを実行 →](https://tiny.cc/custom-charts)

  </TabItem>
  <TabItem value="bar-chart">

`wandb.plot.bar()`

カスタム棒グラフをログします—ラベル付きの値のリストをネイティブに数行で表示します:

```python
data = [[label, val] for (label, val) in zip(labels, values)]
table = wandb.Table(data=data, columns=["label", "value"])
wandb.log(
    {
        "my_bar_chart_id": wandb.plot.bar(
            table, "label", "value", title="Custom Bar Chart"
        )
    }
)
```

任意の棒グラフをログするために使用できます。リストのラベルと値の数が正確に一致する必要があります（つまり、各データポイントには両方が必要です）。

![](@site/static/images/app_ui/line_plot_bar_chart.png)

[アプリ内で確認 →](https://wandb.ai/wandb/plots/reports/Custom-Bar-Charts--VmlldzoyNzExNzk)

[コードを実行 →](https://tiny.cc/custom-charts)

  </TabItem>
  <TabItem value="histogram">

`wandb.plot.histogram()`

カスタムヒストグラムをログします—発生頻度で値をビンに分ける—数行でネイティブに。例えば、予測信頼スコア(`scores`)のリストがあり、それらの分布を視覚化したい場合：

```python
data = [[s] for s in scores]
table = wandb.Table(data=data, columns=["scores"])
wandb.log({"my_histogram": wandb.plot.histogram(table, "scores", title=None)})
```

任意のヒストグラムをログするために使用できます。 `data` はリストのリストであり、行と列の二次元配列をサポートすることを意図しています。

![](/images/app_ui/demo_custom_chart_histogram.png)

[アプリ内で確認 →](https://wandb.ai/wandb/plots/reports/Custom-Histograms--VmlldzoyNzE0NzM)

[コードを実行 →](https://tiny.cc/custom-charts)

  </TabItem>
    <TabItem value="pr-curve">

`wandb.plot.pr_curve()`

[Precision-Recall curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision\_recall\_curve.html#sklearn.metrics.precision\_recall\_curve) を一行で作成します：

```python
plot = wandb.plot.pr_curve(ground_truth, predictions, labels=None, classes_to_plot=None)

wandb.log({"pr": plot})
```

次のような場合にこれをログできます：

* モデルの予測スコア(`predictions`)を一連の例でアクセスできるとき
* それらの例に対する対応する正解ラベル(`ground_truth`)
* （オプション）ラベル/クラス名のリスト（ラベルインデックス0が猫、1が犬、2が鳥などを意味する場合、 `labels=["cat", "dog", "bird"...]`）
* （オプション）プロットで可視化するラベルのサブセット（リスト形式のまま）

![](/images/app_ui/demo_average_precision_lines.png)


[アプリ内で確認 →](https://wandb.ai/wandb/plots/reports/Plot-Precision-Recall-Curves--VmlldzoyNjk1ODY)

[コードを実行 →](https://colab.research.google.com/drive/1mS8ogA3LcZWOXchfJoMrboW3opY1A8BY?usp=sharing)

  </TabItem>
  <TabItem value="roc-curve">

`wandb.plot.roc_curve()`

[ROC curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc\_curve.html#sklearn.metrics.roc\_curve) を一行で作成します：

```python
plot = wandb.plot.roc_curve(
    ground_truth, predictions, labels=None, classes_to_plot=None
)

wandb.log({"roc": plot})
```

次のような場合にこれをログできます：

* モデルの予測スコア(`predictions`)を一連の例でアクセスできるとき
* それらの例に対する対応する正解ラベル(`ground_truth`)
* （オプション）ラベル/クラス名のリスト（ラベルインデックス0が猫、1が犬、2が鳥などを意味する場合、 `labels=["cat", "dog", "bird"...]`）
* （オプション）プロットで可視化するこれらのラベルのサブセット（リスト形式のまま）

![](/images/app_ui/demo_custom_chart_roc_curve.png)

[アプリ内で確認 →](https://wandb.ai/wandb/plots/reports/Plot-ROC-Curves--VmlldzoyNjk3MDE)

[コードを実行 →](https://colab.research.google.com/drive/1\_RMppCqsA8XInV\_jhJz32NCZG6Z5t1RO?usp=sharing)

  </TabItem>
</Tabs>

### カスタムプリセット

組み込みプリセットを調整したり、新しいプリセットを作成したりしてから、チャートを保存します。チャートIDを使用して、スクリプトからそのカスタムプリセットに直接データをログします。

```python
# プロットするための列を含むテーブルを作成
table = wandb.Table(data=data, columns=["step", "height"])

# テーブルの列からチャートのフィールドへのマッピング
fields = {"x": "step", "value": "height"}

# 新しいカスタムチャートプリセットを使ってテーブルを埋める
# 自分の保存済みチャートプリセットを使用するには、vega_spec_nameを変更
my_custom_chart = wandb.plot_table(
    vega_spec_name="carey/new_chart",
    data_table=table,
    fields=fields,
)
```

[コードを実行 →](https://tiny.cc/custom-charts)

![](/images/app_ui/custom_presets.png)

## データをログする

ここでは、スクリプトからログしてカスタムチャートで使用できるデータタイプを紹介します：

* **Config**: 実験の初期設定（独立変数）。トレーニングの開始時に `wandb.config` にキーとしてログされた名前付きフィールドが含まれます（例： `wandb.config.learning_rate = 0.0001`）
* **Summary**: トレーニング中にログされた単一の値（結果または従属変数）、例： `wandb.log({"val_acc" : 0.8})`。トレーニング中に `wandb.log()` を使用してこのキーに複数回書き込むと、サマリーはそのキーの最終値に設定されます。
* **History**: ログされたスカラーの完全な時系列は `history` フィールドを通じてクエリ可能
* **summaryTable**: 複数の値をログする必要がある場合は、 `wandb.Table()` を使用してデータを保存し、カスタムパネルでクエリ
* **historyTable**: 履歴データを表示する必要がある場合はカスタムチャートパネルで `historyTable` をクエリ。 `wandb.Table()` を呼び出すかカスタムチャートをログするたびに、そのステップ用の新しいテーブルが履歴に作成されます。

### カスタムテーブルをログする方法

`wandb.Table()` を使用してデータを二次元配列としてログします。通常、このテーブルの各行は1つのデータポイントを表し、各列はプロットしたい各データポイントの関連フィールド/次元を示します。カスタムパネルを構成するとき、そのテーブル全体は `wandb.log()` に渡される名前付きキー（以下の "custom\_data\_table"）を介してアクセスでき、個別のフィールドは列名（"x"、"y"、"z"）を介してアクセス可能です。実験中の複数のタイムステップでテーブルをログできます。各テーブルの最大サイズは10,000行です。

[Google Colabで試してみる →](https://tiny.cc/custom-charts)

```python
# カスタムデータのテーブルをログ
my_custom_data = [[x1, y1, z1], [x2, y2, z2]]
wandb.log(
    {"custom_data_table": wandb.Table(data=my_custom_data, columns=["x", "y", "z"])}
)
```

## チャートのカスタマイズ

まず新しいカスタムチャートを追加し、次にクエリを編集して可視化するRunsからデータを選択します。クエリは [GraphQL](https://graphql.org) を使用して、Runsのconfig、summary、およびhistoryフィールドからデータを取得します。

![新しいカスタムチャートを追加し、次にクエリを編集](/images/app_ui/customize_chart.gif)

### カスタムビジュアライゼーション

右上の **Chart** を選択してデフォルトのプリセットから始めます。次に、 **Chart fields** を選択して、クエリから取得するデータをチャートの対応フィールドにマッピングします。以下は、クエリからメトリクスを選択し、それを棒グラフのフィールドにマッピングする例です。

![プロジェクト内のRunsの精度を示すカスタム棒グラフの作成](/images/app_ui/demo_make_a_custom_chart_bar_chart.gif)

### Vegaの編集方法

パネル上部の **Edit** をクリックして [Vega](https://vega.github.io/vega/) 編集モードに入ります。ここで、Vega仕様を定義してインタラクティブなチャートを作成できます。[Vega specification](https://vega.github.io/vega/docs/specification/) は、チャートの視覚スタイル（例: タイトル変更、別のカラースキームの選択、ポイントとして曲線を表示するなど）、およびデータ自体を変更するために使用できます（Vega変換を使用して値の配列をヒストグラムにビン分けするなど）。パネルプレビューはインタラクティブに更新されるため、Vega仕様やクエリを編集する際の変更の影響をリアルタイムで確認できます。[Vega documentation and tutorials](https://vega.github.io/vega/) はインスピレーションの素晴らしい源です。

**フィールドリファレンス**

W&Bからデータをチャートに取り込むには、Vega仕様の任意の場所に `"${field:<field-name>}"` の形式のテンプレート文字列を追加します。これにより、右側の **Chart Fields** エリアにドロップダウンが作成され、ユーザーはクエリ結果の列を選択してVegaにマッピングできます。

フィールドのデフォルト値を設定するには、この形式を使用します： `"${field:<field-name>:<placeholder text>}"`

### チャートプリセットの保存

モーダルの下部にあるボタンで特定のビジュアライゼーションパネルに変更を適用します。あるいは、Vega仕様を保存してプロジェクト内の他の場所で使用できます。再利用可能なチャート定義を保存するには、Vegaエディタの上部にある **Save as** をクリックしてプリセットに名前を付けます。

## 記事とガイド

1. [The W&B Machine Learning Visualization IDE](https://wandb.ai/wandb/posts/reports/The-W-B-Machine-Learning-Visualization-IDE--VmlldzoyNjk3Nzg)
2. [Visualizing NLP Attention Based Models](https://wandb.ai/kylegoyette/gradientsandtranslation2/reports/Visualizing-NLP-Attention-Based-Models-Using-Custom-Charts--VmlldzoyNjg2MjM)
3. [Visualizing The Effect of Attention on Gradient Flow](https://wandb.ai/kylegoyette/gradientsandtranslation/reports/Visualizing-The-Effect-of-Attention-on-Gradient-Flow-Using-Custom-Charts--VmlldzoyNjg1NDg)
4. [Logging arbitrary curves](https://wandb.ai/stacey/presets/reports/Logging-Arbitrary-Curves--VmlldzoyNzQyMzA)

## よくある質問

### 近日登場

* **ポーリング**: チャート内のデータを自動更新
* **サンプリング**: 効率を高めるために、パネルに読み込むポイントの総数を動的に調整

### 気を付ける点

* チャートを編集しているときに、期待しているデータがクエリに表示されない場合は、選択したRunsに列がログされていない可能性があります。チャートを保存してRunsテーブルに戻り、 **目** アイコンで可視化したいRunsを選択します。

### カスタムチャートで「ステップスライダー」を表示する方法は？

これはカスタムチャートエディタの「その他の設定」ページで有効にできます。クエリを `summaryTable` の代わりに `historyTable` を使用するように変更すると、カスタムチャートエディタで「ステップセレクタを表示」のオプションが表示されます。これにより、ステップを選択できるスライダーが提供されます。

### カスタムチャートプリセットを削除する方法は？

カスタムチャートエディタに入ります。次に現在選択されているチャートタイプをクリックすると、すべてのプリセットが表示されるメニューが開きます。削除したいプリセットにマウスをホバーし、ゴミ箱アイコンをクリックします。

![](/images/app_ui/delete_custome_chart_preset.gif)

### 共通のユースケース


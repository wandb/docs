---
slug: /guides/app/features/custom-charts
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# カスタムチャート

**カスタムチャート**を使用して、現在のデフォルトのUIでは作成できないチャートを作成します。任意のデータ表をログし、正確にどのように表示したいかを制御できます。[Vega](https://vega.github.io/vega/)の力で、フォント、色、ツールチップの詳細を制御します。

* **可能なこと**：[ローンチアナウンス →](https://wandb.ai/wandb/posts/reports/Announcing-the-W-B-Machine-Learning-Visualization-IDE--VmlldzoyNjk3Nzg)を読む
* **コード**：[ホストされたノートブック →](https://tiny.cc/custom-charts)でライブ例を試す
* **ビデオ**：クイックな[ウォークスルービデオ →](https://www.youtube.com/watch?v=3-N9OV6bkSM)を見る
* **例**：Quick KerasとSklearnの[デモノートブック →](https://colab.research.google.com/drive/1g-gNGokPWM2Qbc8p1Gofud0_5AoZdoSD?usp=sharing)

![vega.github.io/vegaからのサポートされるチャート](/images/app_ui/supported_charts.png)

### 使い方

1. **データを記録**：スクリプトから、[config](../../../../guides/track/config.md)やサマリーデータなどをW&Bと一緒に実行するときと同様に記録します。特定の時間に記録された複数の値のリストを視覚化するには、カスタム `wandb.Table`を使用します。
2. **チャートをカスタマイズ**：[GraphQL](https://graphql.org) クエリを使用して、この記録されたデータの任意の部分を取り込みます。クエリの結果を、強力な視覚化文法である[Vega](https://vega.github.io/vega/)を使って可視化します。
3. **チャートを記録**： `wandb.plot_table()`を使って、スクリプトから自分のプリセットを呼び出します。

![](/images/app_ui/pr_roc.png)

## スクリプトからチャートを記録する

### ビルトインプリセット

これらのプリセットには、ビルトインの `wandb.plot` メソッドがあり、スクリプトから直接チャートを記録し、UIで探している正確な視覚化をすばやく表示できます。
<Tabs
  defaultValue="line-plot"
  values={[
    {label: '折れ線グラフ', value: 'line-plot'},
    {label: '散布図', value: 'scatter-plot'},
    {label: '棒グラフ', value: 'bar-chart'},
    {label: 'ヒストグラム', value: 'histogram'},
    {label: 'PR曲線', value: 'pr-curve'},
    {label: 'ROC曲線', value: 'roc-curve'},
  ]}>
  <TabItem value="line-plot">

`wandb.plot.line()`

カスタム折れ線グラフをログに記録します。接続された順序付きの点（x、y）を任意の軸xとyにプロットします。

```python
data = [[x, y] for (x, y) in zip(x_values, y_values)]
table = wandb.Table(data=data, columns = ["x", "y"])
wandb.log({"my_custom_plot_id" : wandb.plot.line(table, "x", "y", title="Custom Y vs X Line Plot")})
```

任意の2次元で曲線をログに記録することができます。2つの値のリストを互いにプロットする場合、リストの値の数が完全に一致している必要があります（つまり、各点はxとyを持っている必要があります）。

![](/images/app_ui/line_plot.png)

[アプリで見る →](https://wandb.ai/wandb/plots/reports/Custom-Line-Plots--VmlldzoyNjk5NTA)

[コードを実行する →](https://tiny.cc/custom-charts)
</TabItem>
  <TabItem value="scatter-plot">

`wandb.plot.scatter()`

カスタム散布図を記録します。一連の任意の軸xおよびy上の点（x、y）のリストです。

```python
data = [[x, y] for (x, y) in zip(class_x_prediction_scores, class_y_prediction_scores)]
table = wandb.Table(data=data, columns = ["class_x", "class_y"])
wandb.log({"my_custom_id" : wandb.plot.scatter(table, "class_x", "class_y")})
```

これを使用して、任意の2次元上の散布点を記録できます。2つの値のリストを互いにプロットする場合は、リスト内の値の数が正確に一致する必要があります（つまり、各点にはxとyが必要です）。

![](/images/app_ui/demo_scatter_plot.png)

[アプリで見る →](https://wandb.ai/wandb/plots/reports/Custom-Scatter-Plots--VmlldzoyNjk5NDQ)

[コードを実行 →](https://tiny.cc/custom-charts)

  </TabItem>
  <TabItem value="bar-chart">

`wandb.plot.bar()`

カスタムバーチャートを記録します。いくつかの行でネイティブに表示される、ラベル付きの値のリストをバーとして表示します。

```python
data = [[label, val] for (label, val) in zip(labels, values)]
table = wandb.Table(data=data, columns = ["label", "value"])
wandb.log({"my_bar_chart_id" : wandb.plot.bar(table, "label", "value", title="Custom Bar Chart")})
```
これを使用して任意の棒グラフをログに記録できます。リスト内のラベルと値の数が正確に一致しなければならないことに注意してください（つまり、各データポイントには両方が必要です）。

![](@site/static/images/app_ui/line_plot_bar_chart.png)

[アプリで見る →](https://wandb.ai/wandb/plots/reports/Custom-Bar-Charts--VmlldzoyNzExNzk)

[コードを実行する →](https://tiny.cc/custom-charts)

  </TabItem>
  <TabItem value="histogram">

`wandb.plot.histogram()`

カスタムヒストグラムをログに記録し、リスト内の値をカウント/出現頻度でビンに分類して、わずかな行でネイティブに表示します。予測信頼スコア（`scores`）のリストがあり、その分布を視覚化したい場合を考えてみましょう。

```python
data = [[s] for s in scores]
table = wandb.Table(data=data, columns=["scores"])
wandb.log({'my_histogram': wandb.plot.histogram(table, "scores", title=None)})
```

これを使用して任意のヒストグラムをログに記録できます。`data`はリストのリストであり、行と列の2D配列をサポートすることを目的としています。

![](/images/app_ui/demo_custom_chart_histogram.png)

[アプリで見る →](https://wandb.ai/wandb/plots/reports/Custom-Histograms--VmlldzoyNzE0NzM)

[コードを実行する →](https://tiny.cc/custom-charts)

  </TabItem>
    <TabItem value="pr-curve">
`wandb.plot.pr_curve()`

1行で[Precision-Recall curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision\_recall\_curve.html#sklearn.metrics.precision\_recall\_curve) を作成します:

```python
plot = wandb.plot.pr_curve(
    ground_truth, predictions,
    labels=None, classes_to_plot=None)
    
wandb.log({"pr":})
```

以下にアクセスがあるときにこれをログに記録することができます。

* 一連の例に対するモデルの予測スコア（`predictions`）
* それらの例の対応する正解ラベル（`ground_truth`）
* （オプション）ラベル/クラス名のリスト（ラベルインデックス0が猫、1=犬、2=鳥などの場合は、`labels=["cat", "dog", "bird"...]`）
* （オプション）プロットで視覚化するラベルのサブセット（まだリスト形式）

![](/images/app_ui/demo_average_precision_lines.png)

[アプリで見る →](https://wandb.ai/wandb/plots/reports/Plot-Precision-Recall-Curves--VmlldzoyNjk1ODY)

[Run the code →](https://colab.research.google.com/drive/1mS8ogA3LcZWOXchfJoMrboW3opY1A8BY?usp=sharing)

  </TabItem>
  <TabItem value="roc-curve">

`wandb.plot.roc_curve()`
[ROCカーブ](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc\_curve.html#sklearn.metrics.roc\_curve)を1行で作成します。

```python
plot = wandb.plot.roc_curve(
    ground_truth, predictions,
    labels=None, classes_to_plot=None)
    
wandb.log({"roc": plot})
```

以下にアクセスできる場合、いつでもこれをロギングできます。

* 一連の例のモデルの予測スコア（`predictions`）
* それらの例の対応する正解ラベル（`ground_truth`）
* （オプション）ラベル/クラス名のリスト（`labels=["cat", "dog", "bird"...]`）（ラベルインデックス0が猫で1=犬、2=鳥などの場合）
* （オプション）プロットに表示するこれらのラベルのサブセット（リスト形式のまま）

![](/images/app_ui/demo_custom_chart_roc_curve.png)

[アプリで見る →](https://wandb.ai/wandb/plots/reports/Plot-ROC-Curves--VmlldzoyNjk3MDE)

[コードを実行する →](https://colab.research.google.com/drive/1\_RMppCqsA8XInV\_jhJz32NCZG6Z5t1RO?usp=sharing)

  </TabItem>
</Tabs>

### カスタムプリセット

ビルトインプリセットを調整するか、新しいプリセットを作成してから、チャートを保存します。チャートIDを使用して、スクリプトからデータをそのカスタムプリセットに直接ログすることができます。
```python
# プロットするための列を持つテーブルを作成
table = wandb.Table(data=data, columns=["step", "height"])

# テーブルの列からチャートのフィールドにマップする
fields = {"x": "step",
          "value": "height"}

# テーブルを使って新しいカスタムチャートプリセットにデータを入力
# 自分で保存したチャートプリセットを使用するには、vega_spec_nameを変更
my_custom_chart = wandb.plot_table(vega_spec_name="carey/new_chart",
              data_table=table,
              fields=fields,
              )
```

[コードを実行 →](https://tiny.cc/custom-charts)

![](/images/app_ui/custom_presets.png)

## データのログ取り

以下は、スクリプトからログを取得し、カスタムチャートで使用できるデータタイプです。

* **Config**: 実験の初期設定（独立変数）。これには、トレーニングの開始時に`wandb.config`のキーとしてログに記録した名前付きフィールドが含まれます（例：`wandb.config.learning_rate = 0.0001)`）。
* **Summary**: トレーニング中に記録された単一の値（結果や従属変数）。例：`wandb.log({"val_acc" : 0.8})`。トレーニング中に`wandb.log()`でこのキーに複数回書き込むと、サマリーはそのキーの最終値に設定されます。
* **History**: ログされたスカラーの完全な時系列データは、`history`フィールド経由でクエリに利用可能です。
* **summaryTable**: 複数の値のリストをログに記録する必要がある場合は、`wandb.Table()`を使ってデータを保存し、カスタムパネルでそのデータをクエリします。
* **historyTable**: 履歴データを表示する必要がある場合は、カスタムチャートパネルで`historyTable`をクエリしてください。`wandb.Table()`を呼び出したり、カスタムチャートをログに記録するたびに、そのステップの履歴の新しいテーブルが作成されます。
### カスタムテーブルの記録方法

`wandb.Table()`を使用して、データを2次元配列としてログに記録します。通常、このテーブルの各行は1つのデータポイントを表し、各列はプロットするための各データポイントの関連フィールド/次元を示します。カスタムパネルを設定すると、`wandb.log()`に渡された名前付きキー(`"custom_data_table"`以下)でテーブル全体がアクセス可能になり、個々のフィールドが列名("x"、"y"、"z")でアクセス可能になります。実験の途中で複数のタイムステップでテーブルを記録できます。各テーブルの最大サイズは10,000行です。

[Google Colabで試してみる →](https://tiny.cc/custom-charts)

```python
# Logging a custom table of data
my_custom_data = [[x1, y1, z1], [x2, y2, z2]]
wandb.log({"custom_data_table": wandb.Table(data=my_custom_data,
                                columns = ["x", "y", "z"])})
```

## チャートをカスタマイズする

新しいカスタムチャートを追加して始め、クエリを編集して可視化されたランからデータを選択します。クエリは[GraphQL](https://graphql.org)を使用して、実行中のconfig、summary、historyフィールドからデータを取得します。

![新しいカスタムチャートを追加し、クエリを編集する](/images/app_ui/customize_chart.gif)

### カスタムビジュアライゼーション

右上の**Chart**を選択してデフォルトのプリセットから始めます。次に、**Chart fields**を選択して、クエリから取得したデータをチャートの対応するフィールドにマッピングします。以下は、クエリから取得するメトリックを選択し、それを下のバーチャートフィールドにマッピングする例です。

![プロジェクトの実行間での精度を示すカスタムバーチャートを作成する](/images/app_ui/demo_make_a_custom_chart_bar_chart.gif)

### Vegaの編集方法

パネルの上部にある**Edit**をクリックして、[Vega](https://vega.github.io/vega/)編集モードに移ります。ここでは、UIでインタラクティブなチャートを作成する[Vega specification](https://vega.github.io/vega/docs/specification/)を定義できます。チャートの視覚スタイル（例えば、タイトルの変更、異なるカラースキームの選択、曲線を接続された線の代わりに一連の点として表示）からデータ自体（Vega変換を使用して、値の配列をヒストグラムにビン分けするなど）まで、チャートのあらゆる側面を変更できます。パネルプレビューはインタラクティブに更新されるため、Vega仕様やクエリを編集すると、変更の効果がわかりやすくなります。[Vegaのドキュメントとチュートリアル](https://vega.github.io/vega/)は、インスピレーションのすばらしいソースです。

**フィールド参照**
W&Bからチャートにデータを取り込むには、Vega spec内の任意の場所に `"${field:<field-name>}"` という形式のテンプレート文字列を追加してください。これにより、右側の**Chart Fields**エリアにドロップダウンが作成され、ユーザーはそのドロップダウンを使用して、Vegaにマッピングするクエリ結果列を選択できます。

フィールドのデフォルト値を設定するには、次の構文を使用してください：`"${field:<field-name>:<placeholder text>}"`

### チャート設定の保存

変更を特定のビジュアライゼーションパネルに適用するには、モーダルの下部にあるボタンを使用してください。あるいは、Vega specをプロジェクト内の他の箇所で使用するために保存することもできます。再利用可能なチャート定義を保存するには、Vegaエディタの上部にある**Save as**をクリックし、プリセットに名前を付けてください。

## 記事とガイド

1. [W&B機械学習ビジュアライゼーションIDE](https://wandb.ai/wandb/posts/reports/The-W-B-Machine-Learning-Visualization-IDE--VmlldzoyNjk3Nzg)
2. [NLPアテンションベースモデルの可視化](https://wandb.ai/kylegoyette/gradientsandtranslation2/reports/Visualizing-NLP-Attention-Based-Models-Using-Custom-Charts--VmlldzoyNjg2MjM)
3. [アテンション効果の勾配フローを可視化する方法](https://wandb.ai/kylegoyette/gradientsandtranslation/reports/Visualizing-The-Effect-of-Attention-on-Gradient-Flow-Using-Custom-Charts--VmlldzoyNjg1NDg)
4. [任意の曲線のログ記録](https://wandb.ai/stacey/presets/reports/Logging-Arbitrary-Curves--VmlldzoyNzQyMzA)

## よくある質問

### 近日中に追加予定

* **ポーリング**：チャート内のデータの自動更新
* **サンプリング**：パネルに読み込まれるポイントの総数を動的に調整して効率を向上させる

### ゴッチャ

* チャートの編集中に、想定しているデータがクエリに表示されない理由は、選択しているrunに探している列が記録されていないためかもしれません。チャートを保存して、runテーブルに戻り、**eye**アイコンを使用して表示したいrunを選択してください。

### カスタムチャートで「ステップスライダー」を表示するには？

これは、カスタムチャートエディタの「その他の設定」ページで有効にできます。クエリを`summaryTable`から`historyTable`に変更すると、カスタムチャートエディタで「ステップセレクタを表示」するオプションが得られます。これにより、ステップを選択できるスライダーが提供されます。
<!-- ![カスタムチャートでステップスライダーを表示する](/images/app_ui/step_sllider_custon_charts.mov>) -->



### カスタムチャートプリセットを削除する方法は？



これは、カスタムチャートエディタに行くことでできます。次に、現在選択されているチャートタイプをクリックし、プリセットがすべて表示されるメニューを開きます。削除したいプリセットの上にマウスを置き、ゴミ箱アイコンをクリックします。



![](/images/app_ui/delete_custome_chart_preset.gif)





### 一般的なユースケース



* エラーバー付きの棒グラフをカスタマイズする

* カスタムx-y座標が必要なモデル検証メトリクスを表示する（PR曲線のような）

* 二つの異なるモデル/実験からのデータ分布をヒストグラムで重ねて表示する

* トレーニング中に複数のポイントでのメトリックの変化をスナップショットで表示する

* W&Bではまだ利用できない独自の可視化を作成し（そしてできれば世界と共有する）
---
title: カスタム チャート
cascade:
- url: guides/app/features/custom-charts/:filename
menu:
  default:
    identifier: ja-guides-models-app-features-custom-charts-_index
    parent: w-b-app-ui-reference
url: guides/app/features/custom-charts
weight: 2
---

W&B project でカスタムチャートを作成しましょう。任意のデータのテーブルをログして、思い通りに可視化できます。[Vega](https://vega.github.io/vega/) のパワーでフォント、色、ツールチップなど細部までコントロールできます。

* コード: 例の [Colab ノートブック](https://tiny.cc/custom-charts) を試す。
* ビデオ: [ウォークスルー動画](https://www.youtube.com/watch?v=3-N9OV6bkSM) を見る。
* 例: Keras と Sklearn のクイック [デモ ノートブック](https://colab.research.google.com/drive/1g-gNGokPWM2Qbc8p1Gofud0_5AoZdoSD?usp=sharing)

{{< img src="/images/app_ui/supported_charts.png" alt="vega.github.io/vega でサポートされているチャート" max-width="90%" >}}

### 仕組み

1. **Log data**: スクリプトから、[config]({{< relref path="/guides/models/track/config.md" lang="ja" >}}) と summary データをログします。
2. **Customize the chart**: [GraphQL](https://graphql.org) クエリでログ済みデータを取り込みます。強力な可視化文法である [Vega](https://vega.github.io/vega/) でクエリ結果を可視化します。
3. **Log the chart**: スクリプトから `wandb.plot_table()` で自分のプリセットを呼び出します。

{{< img src="/images/app_ui/pr_roc.png" alt="PR と ROC の曲線" >}}

期待したデータが表示されない場合、探している列が選択中の Runs でログされていない可能性があります。チャートを保存してから Runs テーブルに戻り、**目** のアイコンを使って選択中の Runs を確認してください。


## スクリプトからチャートをログする

### ビルトインのプリセット

W&B にはスクリプトから直接ログできるビルトインのチャートプリセットがいくつかあります。これらには折れ線、散布図、棒グラフ、ヒストグラム、PR 曲線、ROC 曲線が含まれます。

{{< tabpane text=true >}}
{{% tab header="折れ線グラフ (Line plot)" value="line-plot" %}}

  `wandb.plot.line()`

  任意の軸 x と y 上の連結された順序付き点 (x, y) のリストとして、カスタムの折れ線グラフをログします。

  ```python
  with wandb.init() as run:
    data = [[x, y] for (x, y) in zip(x_values, y_values)]
    table = wandb.Table(data=data, columns=["x", "y"])
    run.log(
        {
            "my_custom_plot_id": wandb.plot.line(
                table, "x", "y", title="Custom Y vs X Line Plot"
            )
        }
    )
  ```

  折れ線グラフは任意の 2 次元上の曲線をログします。2 つの値のリストを相互にプロットする場合、リスト内の値の数は厳密に一致する必要があります（たとえば各点は x と y の両方を持つ必要があります）。

  {{< img src="/images/app_ui/line_plot.png" alt="カスタム折れ線グラフ" >}}

  [レポート例を見る](https://wandb.ai/wandb/plots/reports/Custom-Line-Plots--VmlldzoyNjk5NTA) あるいは [Google Colab の例を試す](https://tiny.cc/custom-charts)。

{{% /tab %}}

{{% tab header="散布図 (Scatter plot)" value="scatter-plot" %}}

  `wandb.plot.scatter()`

  任意の 2 本の軸 x と y 上の点 (x, y) のリストとして、カスタム散布図をログします。

  ```python
  with wandb.init() as run:
    data = [[x, y] for (x, y) in zip(class_x_prediction_scores, class_y_prediction_scores)]
    table = wandb.Table(data=data, columns=["class_x", "class_y"])
    run.log({"my_custom_id": wandb.plot.scatter(table, "class_x", "class_y")})
  ```

  これは任意の 2 次元上の散布点のログに使えます。2 つの値のリストを相互にプロットする場合、リスト内の値の数は厳密に一致する必要がある点に注意してください（たとえば各点は x と y の両方を持つ必要があります）。

  {{< img src="/images/app_ui/demo_scatter_plot.png" alt="散布図" >}}

  [レポート例を見る](https://wandb.ai/wandb/plots/reports/Custom-Scatter-Plots--VmlldzoyNjk5NDQ) あるいは [Google Colab の例を試す](https://tiny.cc/custom-charts)。

{{% /tab %}}

{{% tab header="棒グラフ (Bar chart)" value="bar-chart" %}}

  `wandb.plot.bar()`

  ラベル付きの値のリストを棒として、数行のコードでネイティブにカスタム棒グラフをログします:

  ```python
  with wandb.init() as run:
    data = [[label, val] for (label, val) in zip(labels, values)]
    table = wandb.Table(data=data, columns=["label", "value"])
    run.log(
        {
            "my_bar_chart_id": wandb.plot.bar(
                table, "label", "value", title="Custom Bar Chart"
            )
        }
    )
  ```

  任意の棒グラフをログできます。ラベルと値のリスト内の数は厳密に一致する必要がある点に注意してください（たとえば各データポイントは両方を持つ必要があります）。

{{< img src="/images/app_ui/demo_bar_plot.png" alt="デモ棒グラフ" >}}

  [レポート例を見る](https://wandb.ai/wandb/plots/reports/Custom-Bar-Charts--VmlldzoyNzExNzk) あるいは [Google Colab の例を試す](https://tiny.cc/custom-charts)。
{{% /tab %}}

{{% tab header="ヒストグラム (Histogram)" value="histogram" %}}

  `wandb.plot.histogram()`

  値のリストを出現回数/頻度でビン分けしてカスタムヒストグラムを、数行のコードでネイティブにログします。たとえば予測の確信度スコア (`scores`) のリストがあり、その分布を可視化したいとします:

  ```python
  with wandb.init() as run:
    data = [[s] for s in scores]
    table = wandb.Table(data=data, columns=["scores"])
    run.log({"my_histogram": wandb.plot.histogram(table, "scores", title=None)})
  ```

  任意のヒストグラムをログできます。`data` はリストのリストで、行と列からなる 2 次元配列を想定している点に注意してください。

  {{< img src="/images/app_ui/demo_custom_chart_histogram.png" alt="カスタムヒストグラム" >}}

  [レポート例を見る](https://wandb.ai/wandb/plots/reports/Custom-Histograms--VmlldzoyNzE0NzM) あるいは [Google Colab の例を試す](https://tiny.cc/custom-charts)。

{{% /tab %}}

{{% tab header="PR 曲線 (PR curve)" value="pr-curve" %}}

  `wandb.plot.pr_curve()`

  1 行で [Precision-Recall 曲線](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html#sklearn.metrics.precision_recall_curve) を作成します:

  ```python
  with wandb.init() as run:
    plot = wandb.plot.pr_curve(ground_truth, predictions, labels=None, classes_to_plot=None)

    run.log({"pr": plot})
  ```

  次の情報にコードからアクセスできるときは、いつでもこれをログできます:

  * 一連のサンプルに対するモデルの予測スコア（`predictions`）
  * それらのサンプルに対応する正解ラベル（`ground_truth`）
  * （任意）ラベル/クラス名のリスト（たとえばラベル 0 = cat、1 = dog、2 = bird であれば `labels=["cat", "dog", "bird"...]`）
  * （任意）プロットで可視化するラベルのサブセット（リスト形式のまま）

  {{< img src="/images/app_ui/demo_average_precision_lines.png" alt="Precision-Recall 曲線" >}}


  [レポート例を見る](https://wandb.ai/wandb/plots/reports/Plot-Precision-Recall-Curves--VmlldzoyNjk1ODY) あるいは [Google Colab の例を試す](https://colab.research.google.com/drive/1mS8ogA3LcZWOXchfJoMrboW3opY1A8BY?usp=sharing)。

{{% /tab %}}

{{% tab header="ROC 曲線 (ROC curve)" value="roc-curve" %}}

  `wandb.plot.roc_curve()`

  1 行で [ROC 曲線](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve) を作成します:

  ```python
  with wandb.init() as run:
    # ground_truth は真のラベルのリスト、predictions は予測スコアのリストです
    ground_truth = [0, 1, 0, 1, 0, 1]
    predictions = [0.1, 0.4, 0.35, 0.8, 0.7, 0.9]

    # ROC 曲線のプロットを作成
    # labels は任意のクラス名リスト、classes_to_plot はそのラベルのうち可視化する任意のサブセットです
    plot = wandb.plot.roc_curve(
        ground_truth, predictions, labels=None, classes_to_plot=None
    )

    run.log({"roc": plot})
  ```

  次の情報にコードからアクセスできるときは、いつでもこれをログできます:

  * 一連のサンプルに対するモデルの予測スコア（`predictions`）
  * それらのサンプルに対応する正解ラベル（`ground_truth`）
  * （任意）ラベル/クラス名のリスト（たとえばラベル 0 = cat、1 = dog、2 = bird であれば `labels=["cat", "dog", "bird"...]`）
  * （任意）これらのラベルのサブセット（リスト形式のまま）をプロットで可視化

  {{< img src="/images/app_ui/demo_custom_chart_roc_curve.png" alt="ROC 曲線" >}}

  [レポート例を見る](https://wandb.ai/wandb/plots/reports/Plot-ROC-Curves--VmlldzoyNjk3MDE) あるいは [Google Colab の例を試す](https://colab.research.google.com/drive/1_RMppCqsA8XInV_jhJz32NCZG6Z5t1RO?usp=sharing)。

{{% /tab %}}
{{< /tabpane >}}

### カスタムプリセット

ビルトインプリセットを調整するか新しく作成してチャートを保存します。チャート ID を使えば、そのカスタムプリセットにスクリプトから直接データをログできます。[Google Colab の例を試す](https://tiny.cc/custom-charts)。

```python
# プロットする列を持つテーブルを作成
table = wandb.Table(data=data, columns=["step", "height"])

# テーブルの列をチャートのフィールドに対応付け
fields = {"x": "step", "value": "height"}

# テーブルを使って新しいカスタムチャートのプリセットにデータを流し込む
# 自分の保存済みチャートプリセットを使うには vega_spec_name を変更
my_custom_chart = wandb.plot_table(
    vega_spec_name="carey/new_chart",
    data_table=table,
    fields=fields,
)
```



{{< img src="/images/app_ui/custom_presets.png" alt="カスタムチャートのプリセット" max-width="90%" >}}

## データをログする

スクリプトから次のデータ型をログし、カスタムチャートで使用できます:

* **Config**: 実験の初期設定（独立変数）。トレーニング開始時に `wandb.Run.config` にキーとしてログした任意のフィールドが含まれます。例: `wandb.Run.config.learning_rate = 0.0001`
* **Summary**: トレーニング中にログする単一値（結果や従属変数）。例: `wandb.Run.log({"val_acc" : 0.8})`。トレーニング中に `wandb.Run.log()` で同じキーに複数回書き込むと、summary はそのキーの最終的な値になります。
* **History**: ログしたスカラーの完全な時系列は、`history` フィールドを介してクエリで利用できます。
* **summaryTable**: 複数の値のリストをログする必要がある場合は、`wandb.Table()` を使ってそのデータを保存し、カスタムパネルでクエリします。
* **historyTable**: 履歴データを確認する必要がある場合は、カスタムチャートパネルで `historyTable` をクエリします。`wandb.Table()` を呼ぶたび、またはカスタムチャートをログするたびに、その step 用の新しいテーブルが history に作成されます。

### カスタムテーブルをログする方法

`wandb.Table()` を使って 2 次元配列としてデータをログします。通常、このテーブルの各行は 1 つのデータポイントを表し、各列はプロットしたい各データポイントの関連フィールド/次元を表します。カスタムパネルを設定すると、テーブル全体は `wandb.Run.log()` に渡したキー（以下では `custom_data_table`）で参照でき、個々のフィールドは列名（`x`、`y`、`z`）で参照できます。テーブルは実験の複数のタイムステップでログできます。各テーブルの最大サイズは 10,000 行です。[Google Colab の例を試す](https://tiny.cc/custom-charts)。



```python
with wandb.init() as run:
  # カスタムデータのテーブルをログ
  my_custom_data = [[x1, y1, z1], [x2, y2, z2]]
  run.log(
      {"custom_data_table": wandb.Table(data=my_custom_data, columns=["x", "y", "z"])}
  )
```

## チャートをカスタマイズ

まず新しいカスタムチャートを追加し、可視の Runs からデータを選択するようにクエリを編集します。クエリは [GraphQL](https://graphql.org) を使って、Runs の config、summary、history フィールドからデータを取得します。

{{< img src="/images/app_ui/customize_chart.gif" alt="カスタムチャートの作成" max=width="90%" >}}

### カスタム可視化

右上の **Chart** を選んでデフォルトのプリセットから開始します。次に **Chart fields** を選び、クエリで取り込むデータをチャート内の対応するフィールドにマッピングします。

次の画像は、メトリクスを選択し、それを下の棒グラフのフィールドにマッピングする方法の例です。

{{< img src="/images/app_ui/demo_make_a_custom_chart_bar_chart.gif" alt="カスタム棒グラフの作成" max-width="90%" >}}

### Vega の編集方法

パネル上部の **Edit** をクリックして [Vega](https://vega.github.io/vega/) の編集モードに入ります。ここでは、UI にインタラクティブなチャートを作成する [Vega specification](https://vega.github.io/vega/docs/specification/) を定義できます。たとえばタイトルを変更したり、別の配色にしたり、曲線を連結した線ではなく点の列として表示したりと、チャートのあらゆる側面を変更できます。また、Vega の transform を使って値の配列をヒストグラムにビン分けするなど、データ自体に変更を加えることもできます。パネルのプレビューはインタラクティブに更新されるため、Vega の spec やクエリを編集しながら変更の効果を確認できます。詳細は [Vega documentation and tutorials ](https://vega.github.io/vega/) を参照してください。

**フィールド参照**

W&B からチャートへデータを取り込むには、Vega の spec の任意の場所に `"${field:<field-name>}"` 形式のテンプレート文字列を追加します。これにより右側の **Chart Fields** 領域にドロップダウンが作成され、ユーザーはクエリ結果の列を選択して Vega にマッピングできます。

フィールドのデフォルト値を設定するには、次の構文を使います: `"${field:<field-name>:<placeholder text>}"`

### チャートプリセットの保存

モーダル下部のボタンで、特定の可視化パネルに変更を適用します。あるいは、Vega の spec を保存して Project の別の場所で使うこともできます。再利用可能なチャート定義を保存するには、Vega エディタ上部の **Save as** をクリックし、プリセットに名前を付けてください。

## 記事とガイド

1. [The W&B Machine Learning Visualization IDE](https://wandb.ai/wandb/posts/reports/The-W-B-Machine-Learning-Visualization-IDE--VmlldzoyNjk3Nzg)
2. [Visualizing NLP Attention Based Models](https://wandb.ai/kylegoyette/gradientsandtranslation2/reports/Visualizing-NLP-Attention-Based-Models-Using-Custom-Charts--VmlldzoyNjg2MjM)
3. [Visualizing The Effect of Attention on Gradient Flow](https://wandb.ai/kylegoyette/gradientsandtranslation/reports/Visualizing-The-Effect-of-Attention-on-Gradient-Flow-Using-Custom-Charts--VmlldzoyNjg1NDg)
4. [Logging arbitrary curves](https://wandb.ai/stacey/presets/reports/Logging-Arbitrary-Curves--VmlldzoyNzQyMzA)


## よくあるユースケース

* 誤差バー付きの棒グラフをカスタマイズ
* カスタムの x-y 座標が必要なモデルの検証メトリクスを表示（PR 曲線など）
* 2 つの異なるモデル/実験のデータ分布をヒストグラムで重ねて表示
* トレーニング中の複数の時点でのスナップショットを通してメトリクスの変化を示す
* W&B にまだない独自の可視化を作成する（そしてぜひ世界と共有する）
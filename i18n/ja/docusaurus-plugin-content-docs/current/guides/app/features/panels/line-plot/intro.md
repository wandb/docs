---
slug: /guides/app/features/panels/line-plot
description: Visualize metrics, customize axes, and compare multiple lines on the same plot
displayed_sidebar: ja
---

# 折れ線グラフ

折れ線グラフは、**wandb.log()**を使用して時間経過でメトリクスをプロットすると、デフォルトで表示されます。チャートの設定をカスタマイズして、同じプロット上で複数の線を比較し、カスタム軸を計算し、ラベルの名前を変更します。

![](/images/app_ui/line_plot_example.png)

## 設定

**データ**

* **X軸**： ステップと相対時間を含むデフォルトのx軸を選択するか、カスタムx軸を選択します。カスタムx軸を使用したい場合は、y軸を記録するのと同じ`wandb.log()`の呼び出しでそれを記録してください。
  * **相対時間（ウォール）**は、プロセスが開始されてからの経過時間です。つまり、ランを開始してから1日後に再開して何かをログインした場合、24時間後にプロットされます。
  * **相対時間（プロセス）**は、実行中のプロセス内の時間であり、ランを開始して10秒間実行し、1日後に再開した場合、その点は10秒でプロットされます
  * **ウォールタイム**は、グラフ上の最初のランの開始時刻から経過した分数です
  * **ステップ**は、デフォルトで`wandb.log()`が呼び出されるたびに増加し、モデルからのトレーニングステップの数を反映することを目的としています
* **Y軸**： 時間経過で変化するメトリクスとハイパーパラメータを含む、ログされた値からy軸を選択
* **最小、最大、および対数スケール**： 折れ線グラフのx軸およびy軸の最小、最大、および対数スケール設定
* **平滑化および外れ値の除外**： 折れ線グラフの平滑化を変更するか、デフォルトのプロット最小および最大スケールから外れ値を除外するために再スケーリングを行います
* **表示する最大ラン数**： この数値を増やして、一度に折れ線グラフに表示する線の数を増やします。デフォルトでは10個のランが表示されます。10個以上のランが利用可能な場合でも、チャートが表示される数を制限している場合は、「最初の10個のランを表示」というメッセージがチャートの上部に表示されます。
* **チャートタイプ**： 折れ線グラフ、エリアグラフ、およびパーセンテージエリアグラフの間で変更できます

**X軸の設定**
x軸は、グラフレベルだけでなく、プロジェクトページやレポートページ全体でも設定できます。以下は、グローバル設定の見た目です。

![](/images/app_ui/x_axis_global_settings.png)
:::info
折れ線グラフの設定で**複数のy軸**を選択して、例えば精度と検証精度のような異なる指標を同じチャートで比較できます。
:::

**グルーピング**

* グルーピングをオンにして、平均値を可視化する設定を表示します。
* **グループキー**: 列を選択し、その列で同じ値を持つすべてのrunがまとめられます。
* **集計**: グラフ上の線の値。オプションは、グループの平均、中央値、最小値、最大値です。
* **範囲**: グループ化された曲線の背後にある影の領域の動作を切り替えます。Noneは影の領域がないことを意味します。Min/Maxは、グループ内のポイントの全範囲をカバーする影の領域を示します。Std Devは、グループ内の値の標準偏差を示します。Std Errは、標準誤差を影の領域として示します。
* **サンプルされたrun**: 選択したrunが何百もある場合、デフォルトでは最初の100だけをサンプリングします。すべてのrunをグループ化の計算に含めるように選択することができますが、UIでの動作が遅くなる可能性があります。

**凡例**

* **タイトル**: 折れ線グラフのカスタムタイトルを追加し、チャートの上部に表示されます。
* **X軸のタイトル**: 折れ線グラフのx軸のカスタムタイトルを追加し、チャートの右下隅に表示されます。
* **Y軸のタイトル**: 折れ線グラフのy軸のカスタムタイトルを追加し、チャートの左上隅に表示されます。
* **凡例**: 各線のプロットの凡例に表示したい項目を選択します。例えば、run名と学習率を表示することができます。
* **凡例テンプレート**: 完全にカスタマイズ可能で、折れ線グラフの上部に表示されるテンプレートや、プロットにマウスをオーバーした時に表示される凡例に、正確にどのテキストや変数を表示するかを指定できる強力なテンプレートです。

![Editing the line plot legend to show hyperparameters](/images/app_ui/legend.png)

**式**

* **Y軸の式**: グラフに計算された指標を追加します。ログされた指標やハイパーパラメータなどの設定値を使用してカスタム線を計算できます。
* **X軸の式**: カスタム式を使用して計算された値でx軸をリスケールします。役に立つ変数には、デフォルトのx軸である\*\*\_step\*\*や、サマリー値を参照するための構文 `${summary:value}` があります。

## プロット上で平均値を可視化する

複数の実験があり、その値の平均をプロットで見たい場合は、テーブル内のグルーピング機能を使用できます。runテーブルの上部にある「グループ」をクリックし、すべてのグループ化された値をグラフに表示するために、「すべて」を選択してください。
グラフは平均化する前に以下のようになります。

![](/images/app_ui/demo_precision_lines.png)

ここでは、実行間で平均値を見るために線をグループ化しました。

![](/images/app_ui/demo_average_precision_lines.png)

## プロット上のNaN値を可視化する

`wandb.log`を用いて、PyTorchテンソルを含む`NaN`値を折れ線グラフにプロットすることもできます。例えば、以下のように表現できます。

```python
wandb.log({"test": [..., float("nan"), ...]})
```

![](/images/app_ui/visualize_nan.png)

## 1つのチャートで2つのメトリクスを比較する

実行をクリックして実行ページに移動します。これはStaceyのEstuaryプロジェクトからの[サンプル実行](https://app.wandb.ai/stacey/estuary/runs/9qha4fuu?workspace=user-carey)です。自動生成されたチャートは単一のメトリクスを表示しています。

![](@site/static/images/app_ui/visualization_add.png)

ページの右上にある**プラス記号**をクリックし、**Line Plot**を選択します。

![](https://downloads.intercomcdn.com/i/o/142936481/d0648728180887c52ab46549/image.png)

**Y変数**フィールドで、比較したいいくつかのメトリクスを選択します。それらは折れ線グラフ上にまとめて表示されます。
![](https://downloads.intercomcdn.com/i/o/146033909/899fc05e30795a1d7699dc82/Screen+Shot+2019-09-04+at+9.10.52+AM.png)

## 折れ線グラフの色を変更する

時には、runsのデフォルトの色が比較に役立たないことがあります。それを解決するために、wandbは手動で色を変更できる2つのインスタンスを提供しています。

### runテーブルから

各runは、初期化時にデフォルトでランダムな色が割り当てられます。

![runsにランダムに割り当てられた色](/images/app_ui/line_plots_run_table_random_colors.png)

色のいずれかをクリックすると、手動で選択したい色を選べるカラーパレットが表示されます。

![カラーパレット](/images/app_ui/line_plots_run_table_color_palette.png)

### チャートの凡例設定から

チャートの凡例設定からも、runsの色を変更することができます。

![](/images/app_ui/plot_style_line_plot_legend.png)

## 異なるx軸で視覚化する

実験にかかった絶対時間を見たり、実験が実行された日を確認したい場合は、x軸を切り替えることができます。ここでは、ステップから相対時間に切り替え、次に壁掛け時間に切り替える例を示します。

![](/images/app_ui/howto_use_relative_time_or_wall_time.gif)
## エリアプロット



折れ線グラフの設定で、高度なタブを開いて、異なるプロットスタイルをクリックすると、エリアプロットやパーセンテージエリアプロットが得られます。



![](/images/app_ui/line_plots_area_plots.gif)



## ズーム



矩形をクリックしてドラッグすると、縦方向と横方向に同時にズームできます。これにより、x軸とy軸のズームが変更されます。



![](/images/app_ui/line_plots_zoom.gif)



## チャート凡例を非表示にする



この簡単なトグルで折れ線グラフの凡例をオフにします：



![](/images/app_ui/demo_hide_legend.gif)
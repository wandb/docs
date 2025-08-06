---
title: 'line_series()

  '
data_type_classification: function
menu:
  reference:
    identifier: ja-ref-python-sdk-custom-charts-line_series
object_type: python_sdk_custom_charts
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/plot/line_series.py >}}




### <kbd>関数</kbd> `line_series`

```python
line_series(
    xs: 'Iterable[Iterable[Any]] | Iterable[Any]',
    ys: 'Iterable[Iterable[Any]]',
    keys: 'Iterable[str] | None' = None,
    title: 'str' = '',
    xname: 'str' = 'x',
    split_table: 'bool' = False
) → CustomChart
```

折れ線の系列チャートを作成します。



**引数:**
 
 - `xs`:  x 値のシーケンス。単一の配列が指定された場合、すべての y 値がその x 配列に対してプロットされます。配列の配列が指定された場合は、それぞれの y 値が対応する x 配列に対してプロットされます。
 - `ys`:  y 値のシーケンス。それぞれのイテラブルが個別の折れ線系列を表します。
 - `keys`:  各折れ線系列にラベル付けするためのキーのシーケンス。指定しない場合、自動的に "line_1"、"line_2" のように生成されます。
 - `title`:  チャートのタイトル。
 - `xname`:  x 軸のラベル。
 - `split_table`:  テーブルを W&B UI 内で別セクションに分割するかどうか。`True` の場合、「Custom Chart Tables」というセクションで表示されます。デフォルトは `False` です。



**戻り値:**
 
 - `CustomChart`:  W&B に ログできるカスタムチャートオブジェクト。チャートをログするには `wandb.log()` に渡してください。



**使用例:**
 すべての y 系列が同じ x 値に対してプロットされる、単一の x 配列をログする場合:

```python
import wandb

# W&B run を初期化
with wandb.init(project="line_series_example") as run:
    # すべての y 系列で共有する x 値
    xs = list(range(10))

    # プロットする複数の y 系列
    ys = [
         [i for i in range(10)],  # y = x
         [i**2 for i in range(10)],  # y = x^2
         [i**3 for i in range(10)],  # y = x^3
    ]

    # 折れ線系列チャートを生成してログする
    line_series_chart = wandb.plot.line_series(
         xs,
         ys,
         title="title",
         xname="step",
    )
    run.log({"line-series-single-x": line_series_chart})
```

この例では、単一の `xs` シリーズ（共有された x 値）がすべての `ys` シリーズに使われます。その結果、各 y 系列は同じ x 値（0-9）に対してプロットされます。

各 y 系列が自分専用の x 配列に対してプロットされる、複数の x 配列をログする場合:

```python
import wandb

# W&B run を初期化
with wandb.init(project="line_series_example") as run:
    # 各 y 系列ごとに異なる x 値
    xs = [
         [i for i in range(10)],  # 最初の系列の x
         [2 * i for i in range(10)],  # 2番目の系列（引き伸ばし）
         [3 * i for i in range(10)],  # 3番目の系列（さらに引き伸ばし）
    ]

    # 対応する y 系列
    ys = [
         [i for i in range(10)],  # y = x
         [i**2 for i in range(10)],  # y = x^2
         [i**3 for i in range(10)],  # y = x^3
    ]

    # 折れ線系列チャートを生成してログする
    line_series_chart = wandb.plot.line_series(
         xs, ys, title="Multiple X Arrays Example", xname="Step"
    )
    run.log({"line-series-multiple-x": line_series_chart})
```

この例では、各 y 系列ごとに専用の x 系列を使用してプロットします。データ系列ごとに x 値が異なる場合など、より柔軟に扱うことができます。

`keys` を使った折れ線ラベルのカスタマイズ例:

```python
import wandb

# W&B run を初期化
with wandb.init(project="line_series_example") as run:
    xs = list(range(10))  # 単一の x 配列
    ys = [
         [i for i in range(10)],  # y = x
         [i**2 for i in range(10)],  # y = x^2
         [i**3 for i in range(10)],  # y = x^3
    ]

    # 各折れ線のカスタムラベル
    keys = ["Linear", "Quadratic", "Cubic"]

    # 折れ線系列チャートを生成してログする
    line_series_chart = wandb.plot.line_series(
         xs,
         ys,
         keys=keys,  # カスタムキー（折れ線ラベル）
         title="Custom Line Labels Example",
         xname="Step",
    )
    run.log({"line-series-custom-keys": line_series_chart})
```

この例では、`keys` 引数を使って折れ線のラベルを自由に指定する方法を示しています。ラベルは凡例で "Linear"、"Quadratic"、"Cubic" と表示されます。
---
title: 'line_series()

  '
object_type: python_sdk_custom_charts
data_type_classification: function
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

折れ線グラフ（ラインシリーズチャート）を作成します。

**引数:**
 
 - `xs`:  x 値のシーケンス。1つの配列を渡すと、すべての y 値がその x 配列に対してプロットされます。複数の配列の配列を渡す場合は、各 y 値が対応する x 配列に対してプロットされます。
 - `ys`:  y 値のシーケンス。各イテラブルが個別のラインシリーズ（折れ線）となります。
 - `keys`:  各ラインシリーズをラベル付けするためのキー群。指定しない場合は "line_1", "line_2" のように自動生成されます。
 - `title`:  チャートのタイトル。
 - `xname`:  x 軸のラベル。
 - `split_table`:  テーブルを W&B UI 内で別のセクションに分けて表示するかどうか。`True` の場合、「Custom Chart Tables」というセクションにテーブルが表示されます。デフォルトは `False` です。

**戻り値:**
 
 - `CustomChart`:  W&B にログできるカスタムチャートオブジェクト。チャートをログするには `wandb.log()` に渡してください。

**使用例:**
単一の x 配列で、すべての y シリーズが同じ x 値に対してプロットされる例：

```python
import wandb

# W&B run を初期化
with wandb.init(project="line_series_example") as run:
    # すべての y シリーズで共有する x 値
    xs = list(range(10))

    # プロットする複数の y シリーズ
    ys = [
         [i for i in range(10)],  # y = x
         [i**2 for i in range(10)],  # y = x^2
         [i**3 for i in range(10)],  # y = x^3
    ]

    # ラインシリーズチャートを生成し、ログ
    line_series_chart = wandb.plot.line_series(
         xs,
         ys,
         title="title",
         xname="step",
    )
    run.log({"line-series-single-x": line_series_chart})
```

この例では、単一の `xs`（共通 x 値）がすべての `ys` シリーズに使用されています。つまり、全ての y シリーズが x=0〜9 に対して描画されます。

各 y シリーズに専用の x 配列を用いる場合の例：

```python
import wandb

# W&B run を初期化
with wandb.init(project="line_series_example") as run:
    # 各 y シリーズごとに異なる x 値
    xs = [
         [i for i in range(10)],  # 1本目のシリーズの x
         [2 * i for i in range(10)],  # 2本目（より広がる）
         [3 * i for i in range(10)],  # 3本目（さらに広がる）
    ]

    # 対応する y シリーズ
    ys = [
         [i for i in range(10)],  # y = x
         [i**2 for i in range(10)],  # y = x^2
         [i**3 for i in range(10)],  # y = x^3
    ]

    # ラインシリーズチャートを生成し、ログ
    line_series_chart = wandb.plot.line_series(
         xs, ys, title="Multiple X Arrays Example", xname="Step"
    )
    run.log({"line-series-multiple-x": line_series_chart})
```

この例では、各 y シリーズごとに独自の x 配列が利用されます。これにより、x がデータ系列ごとに均一でない場合でも柔軟に可視化できます。

`keys` でラベルをカスタマイズする例：

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

    # 各ラインのカスタムラベル
    keys = ["Linear", "Quadratic", "Cubic"]

    # ラインシリーズチャートを生成し、ログ
    line_series_chart = wandb.plot.line_series(
         xs,
         ys,
         keys=keys,  # カスタムキー（ラインラベル）
         title="Custom Line Labels Example",
         xname="Step",
    )
    run.log({"line-series-custom-keys": line_series_chart})
```

この例では、`keys` 引数を用いて各ライン（折れ線）のラベルをカスタマイズする方法を示しています。ラベルは凡例に "Linear", "Quadratic", "Cubic" として表示されます。
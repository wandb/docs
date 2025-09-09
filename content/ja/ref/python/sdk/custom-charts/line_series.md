---
title: line_series()
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

ライン シリーズ チャートを作成します。 



**引数:**
 
 - `xs`:  x の 値 のシーケンス。単一の配列を渡すと、すべての y 値がその x 配列に対してプロットされます。配列の配列を渡すと、各 y 値は対応する x 配列に対してプロットされます。 
 - `ys`:  y の 値 のシーケンス。各イテラブルが個別のライン シリーズを表します。 
 - `keys`:  各ライン シリーズにラベルを付けるための キー のシーケンス。指定しない場合は "line_1"、"line_2" のように自動生成されます。 
 - `title`:  チャートのタイトル。 
 - `xname`:  x 軸のラベル。 
 - `split_table`:  テーブル を W&B UI の別セクションに分けて表示するかどうか。`True` の場合、テーブルは "Custom Chart Tables" という名前のセクションに表示されます。既定値は `False`。 



**戻り値:**
 
 - `CustomChart`:  W&B に ログ できるカスタム チャート オブジェクト。チャートを ログ するには `wandb.log()` に渡してください。 



**例:**
 すべての y シリーズが同じ x 値に対してプロットされる、単一の x 配列を ログ する例: 

```python
import wandb

# W&B の run を初期化
with wandb.init(project="line_series_example") as run:
    # すべての y シリーズで共有される x 値
    xs = list(range(10))

    # 描画する複数の y シリーズ
    ys = [
         [i for i in range(10)],  # y = x
         [i**2 for i in range(10)],  # y = x^2
         [i**3 for i in range(10)],  # y = x^3
    ]

    # ライン シリーズ チャートを生成して ログ
    line_series_chart = wandb.plot.line_series(
         xs,
         ys,
         title="title",
         xname="step",
    )
    run.log({"line-series-single-x": line_series_chart})
``` 

この例では、単一の `xs` シリーズ (共有の x 値) をすべての `ys` シリーズに使っています。各 y シリーズは同じ x 値 (0-9) に対してプロットされます。 

各 y シリーズが対応する x 配列に対してプロットされる、複数の x 配列を ログ する例: 

```python
import wandb

# W&B の run を初期化
with wandb.init(project="line_series_example") as run:
    # 各 y シリーズ用の個別の x 値
    xs = [
         [i for i in range(10)],  # 1 本目のシリーズの x
         [2 * i for i in range(10)],  # 2 本目のシリーズの x (伸長)
         [3 * i for i in range(10)],  # 3 本目のシリーズの x (さらに伸長)
    ]

    # 対応する y シリーズ
    ys = [
         [i for i in range(10)],  # y = x
         [i**2 for i in range(10)],  # y = x^2
         [i**3 for i in range(10)],  # y = x^3
    ]

    # ライン シリーズ チャートを生成して ログ
    line_series_chart = wandb.plot.line_series(
         xs, ys, title="Multiple X Arrays Example", xname="Step"
    )
    run.log({"line-series-multiple-x": line_series_chart})
``` 

この例では、各 y シリーズがそれぞれ固有の x シリーズに対してプロットされます。x 値がデータ シリーズ間で一様でない場合に、より柔軟に対応できます。 

`keys` を使って線のラベルをカスタマイズ: 

```python
import wandb

# W&B の run を初期化
with wandb.init(project="line_series_example") as run:
    xs = list(range(10))  # 単一の x 配列
    ys = [
         [i for i in range(10)],  # y = x
         [i**2 for i in range(10)],  # y = x^2
         [i**3 for i in range(10)],  # y = x^3
    ]

    # 各線のカスタム ラベル
    keys = ["Linear", "Quadratic", "Cubic"]

    # ライン シリーズ チャートを生成して ログ
    line_series_chart = wandb.plot.line_series(
         xs,
         ys,
         keys=keys,  # カスタム キー (線のラベル)
         title="Custom Line Labels Example",
         xname="Step",
    )
    run.log({"line-series-custom-keys": line_series_chart})
``` 

この例では、`keys` 引数を使って線のラベルをカスタム指定する方法を示しています。キーは凡例に "Linear"、"Quadratic"、"Cubic" と表示されます。
---
title: ヒストグラム
menu:
  reference:
    identifier: ja-ref-python-data-types-histogram
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/histogram.py#L18-L94 >}}

wandb のヒストグラム用クラス。

```python
Histogram(
    sequence: Optional[Sequence] = None,
    np_histogram: Optional['NumpyHistogram'] = None,
    num_bins: int = 64
) -> None
```

このオブジェクトは numpy のヒストグラム関数と同様に動作します。
https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html

#### 例:

シーケンスからヒストグラムを生成

```python
wandb.Histogram([1, 2, 3])
```

np.histogram から効率的に初期化。

```python
hist = np.histogram(data)
wandb.Histogram(np_histogram=hist)
```

| Args |  |
| :--- | :--- |
|  `sequence` |  (array_like) ヒストグラムの入力データ |
|  `np_histogram` |  (numpy histogram) あらかじめ計算されたヒストグラムの代替入力 |
|  `num_bins` |  (int) ヒストグラムのビンの数。デフォルトのビンの数は 64 です。ビンの最大数は 512 です |

| Attributes |  |
| :--- | :--- |
|  `bins` |  ([float]) ビンの境界 |
|  `histogram` |  ([int]) 各ビンに入る要素の数 |

| Class Variables |  |
| :--- | :--- |
|  `MAX_LENGTH`<a id="MAX_LENGTH"></a> |  `512` |
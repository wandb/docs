# ヒストグラム

[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)GitHubでソースを表示する](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/data_types/histogram.py#L17-L95)

wandbのヒストグラム用クラスです。

```python
Histogram(
 sequence: Optional[Sequence] = None,
 np_histogram: Optional['NumpyHistogram'] = None,
 num_bins: int = 64
) -> None
```

このオブジェクトは、numpyのヒストグラム関数と同様に動作します。
https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html

#### 例:

シーケンスからヒストグラムを生成する
```python
wandb.Histogram([1, 2, 3])
```
np.histogram から効率的に初期化する方法。
```python
hist = np.histogram(data)
wandb.Histogram(np_histogram=hist)
```



| 引数 | |
| :--- | :--- |
| `sequence` | (array_like) ヒストグラムの入力データ |
| `np_histogram` | (numpy histogram) 事前に計算されたヒストグラムの代替入力 |
| `num_bins` | (int) ヒストグラムのビンの数。デフォルトのビンの数は64。ビンの最大数は512 |





| 属性 | |
| :--- | :--- |
| `bins` | ([float]) ビンのエッジ |
| `histogram` | ([int]) 各ビンに分類される要素の数 |





| クラス変数 | |
| :--- | :--- |
| `MAX_LENGTH` | `512` |
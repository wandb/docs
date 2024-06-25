
# ヒストグラム

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/data_types/histogram.py#L18-L96' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHubでソースコードを見る</a></button></p>

wandb のヒストグラム用クラス。

```python
Histogram(
    sequence: Optional[Sequence] = None,
    np_histogram: Optional('NumpyHistogram') = None,
    num_bins: int = 64
) -> None
```

このオブジェクトは numpy のヒストグラム機能と同様に動作します
https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html

#### 例:

シーケンスからヒストグラムを生成する

```python
wandb.Histogram([1, 2, 3])
```

np.histogram から効率的に初期化する。

```python
hist = np.histogram(data)
wandb.Histogram(np_histogram=hist)
```

| 引数 |  |
| :--- | :--- |
|  `sequence` |  (array_like) ヒストグラムの入力データ |
|  `np_histogram` |  (numpy ヒストグラム) 事前に計算されたヒストグラムの別の入力 |
|  `num_bins` |  (int) ヒストグラムのビンの数。デフォルトのビンの数は64。ビンの最大数は512 |

| 属性 |  |
| :--- | :--- |
|  `bins` |  ([float]) ビンの境界 |
|  `histogram` |  ([int]) 各ビンに属する要素の数 |

| クラス変数 |  |
| :--- | :--- |
|  `MAX_LENGTH`<a id="MAX_LENGTH"></a> |  `512` |
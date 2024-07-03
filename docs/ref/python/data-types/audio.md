# Audio

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/data_types.py#L982-L1126' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

Wandb クラスのオーディオクリップ。

```python
Audio(
    data_or_path, sample_rate=None, caption=None
)
```

| 引数 |  |
| :--- | :--- |
|  `data_or_path` |  (string または numpy array) オーディオファイルへのパス、またはオーディオデータの numpy 配列。 |
|  `sample_rate` |  (int) サンプルレート。生の numpy 配列のオーディオデータを渡す場合に必要。 |
|  `caption` |  (string) オーディオに表示するキャプション。 |

## メソッド

### `durations`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/data_types.py#L1084-L1086)

```python
@classmethod
durations(
    audio_list
)
```

### `resolve_ref`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/data_types.py#L1100-L1112)

```python
resolve_ref()
```

### `sample_rates`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/data_types.py#L1088-L1090)

```python
@classmethod
sample_rates(
    audio_list
)
```
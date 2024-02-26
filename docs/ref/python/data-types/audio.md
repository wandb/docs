# オーディオ

[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)GitHubでソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/data_types.py#L1042-L1186)

Wandbのオーディオクリップ用クラス。

```python
Audio(
 データまたはパス, サンプルレート=None, キャプション=None
)
```

| 引数 | |
| :--- | :--- |
| `data_or_path` | (文字列またはnumpy配列) オーディオファイルへのパスまたはオーディオデータのnumpy配列。 |
| `sample_rate` | (int) サンプルレート。raw numpy配列のオーディオデータを渡すときに必要です。 |
| `caption` | (文字列) オーディオに表示するキャプション。 |

## メソッド
### `durations`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/data_types.py#L1144-L1146)

```python
@classmethod
durations(
 audio_list
)
```




### `resolve_ref`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/data_types.py#L1160-L1172)

```python
resolve_ref()
```




### `sample_rates`

[ソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/data_types.py#L1148-L1150)

```python

@classmethod

sample_rates(

 audio_list

)

```
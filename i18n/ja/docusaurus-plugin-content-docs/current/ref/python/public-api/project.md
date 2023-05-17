# プロジェクト

[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)GitHubでソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L1465-L1547)

プロジェクトは、runsのための名前空間です。

```python
Project(
 client, entity, project, attrs
)
```

| 属性 | |
| :--- | :--- |

## メソッド

### `artifacts_types`

[ソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L1498-L1500)

```python
artifacts_types(
 per_page=50
)
```




### `display`



[ソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L979-L990)

```python
display(
 height=420, hidden=(False)
) -> bool
```

このオブジェクトをjupyterで表示します。

### `snake_to_camel`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L975-L977)

```python
snake_to_camel(
 string
)
```




### `sweeps`



[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L1502-L1547)

```python
sweeps()
```




### `to_html`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L1482-L1490)

```python
to_html(

 height=420, hidden=(False)

)
```

このプロジェクトを表示するiframeを含むHTMLを生成します。
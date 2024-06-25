
# Html

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/data_types/html.py#L18-L107' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>


任意のHTML用Wandbクラス。

```python
Html(
    data: Union[str, 'TextIO'],
    inject: bool = (True)
) -> None
```

| 引数 |  |
| :--- | :--- |
|  `data` |  (文字列またはioオブジェクト) wandbに表示するHTML |
|  `inject` |  (ブール値) HTMLオブジェクトにスタイルシートを追加します。Falseに設定すると、HTMLは変更されずにそのまま通過します。 |

## メソッド

### `inject_head`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/data_types/html.py#L59-L74)

```python
inject_head() -> None
```
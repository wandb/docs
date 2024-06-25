
# Project

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/projects.py#L79-L160' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

プロジェクトは run のための名前空間です。

```python
Project(
    client, entity, project, attrs
)
```

| 属性 |  |
| :--- | :--- |

## メソッド

### `artifacts_types`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/projects.py#L112-L114)

```python
artifacts_types(
    per_page=50
)
```

### `display`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/attrs.py#L15-L26)

```python
display(
    height=420, hidden=(False)
) -> bool
```

jupyter でこの オブジェクト を表示します。

### `snake_to_camel`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/attrs.py#L11-L13)

```python
snake_to_camel(
    string
)
```

### `sweeps`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/projects.py#L116-L160)

```python
sweeps()
```

### `to_html`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/projects.py#L96-L104)

```python
to_html(
    height=420, hidden=(False)
)
```

このプロジェクトを表示する iframe を含む HTML を生成します。
# File

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/files.py#L108-L195' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

Fileはwandbによって保存されたファイルに関連するクラスです。

```python
File(
    client, attrs
)
```

| 属性 |  |
| :--- | :--- |

## メソッド

### `delete`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/files.py#L175-L188)

```python
delete()
```

### `display`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/attrs.py#L15-L26)

```python
display(
    height=420, hidden=(False)
) -> bool
```

このオブジェクトをjupyterで表示します。

### `download`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/files.py#L134-L173)

```python
download(
    root: str = ".",
    replace: bool = (False),
    exist_ok: bool = (False),
    api: Optional[Api] = None
) -> io.TextIOWrapper
```

wandbサーバーからrunによって以前保存されたファイルをダウンロードします。

| 引数 |  |
| :--- | :--- |
|  replace (boolean): `True`の場合、ダウンロードは既存のローカルファイルを上書きします。デフォルトは`False`。 root (str): ローカルディレクトリーにファイルを保存します。デフォルトは "."。 exist_ok (boolean): `True`の場合、既にファイルが存在していてもValueErrorを発生させず、replace=Trueでない限り再ダウンロードしません。デフォルトは`False`。 api (Api, optional): 提供された場合、ファイルをダウンロードするために使用される`Api`インスタンス。 |

| 発生するエラー |  |
| :--- | :--- |
|  ファイルが既に存在し、replace=Falseかつexist_ok=Falseの場合に`ValueError`を発生させます。 |

### `snake_to_camel`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/attrs.py#L11-L13)

```python
snake_to_camel(
    string
)
```

### `to_html`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/attrs.py#L28-L29)

```python
to_html(
    *args, **kwargs
)
```
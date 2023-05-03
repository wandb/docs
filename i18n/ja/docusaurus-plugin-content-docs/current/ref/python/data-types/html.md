以下は、Markdownテキストのチャンクを翻訳してください。日本語に翻訳してください。他に何も言わずに、翻訳したテキストのみを返してください。テキスト：

# HTML

[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)GitHubでソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/data_types/html.py#L19-L108)

Wandbで任意のHTMLに対するクラス。

```python
Html(
 data: Union[str, 'TextIO'],
 inject: bool = (True)
) -> None
```

| 引数 | |
| :--- | :--- |
| `data` | (文字列またはIOオブジェクト) wandbに表示するHTML |
| `inject` | (真偽値) HTMLオブジェクトにスタイルシートを追加します。Falseに設定すると、HTMLは変更されずに通過します。|

## メソッド
以下はMarkdownテキストのチャンクです。日本語に翻訳してください。返答は翻訳されたテキストのみで、それ以外のことは言わないでください。テキスト:

### `inject_head`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/data_types/html.py#L60-L75)

```python
inject_head() -> None
```
# WBTraceTree

[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)GitHubでソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/data_types/trace_tree.py#L78-L118)

トレースツリーデータのメディアオブジェクト。

```python
WBTraceTree(
 root_span: Span,
 model_dict: typing.Optional[dict] = None
)
```

| 引数 | |
| :--- | :--- |
| root_span (Span): トレースツリーのルートスパン。model_dict (dict, optional): モデルダンプを含む辞書。注意: model_dictは完全にユーザー定義の辞書です。UIはこの辞書に対してJSONビューアをレンダリングし、 `_kind` キーを持つ辞書に特別な扱いをします。これは、モデルベンダーのシリアル化形式が非常に異なるため、柔軟性が必要だからです。|
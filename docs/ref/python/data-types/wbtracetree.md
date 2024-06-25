
# WBTraceTree

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/data_types/trace_tree.py#L80-L119' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

トレースツリーのデータのメディアオブジェクト。

```python
WBTraceTree(
    root_span: Span,
    model_dict: typing.Optional[dict] = None
)
```

| 引数 |  |
| :--- | :--- |
| root_span (Span): トレースツリーのルートスパン。 model_dict (dict, optional): モデルダンプを含む辞書。注意: model_dict は完全にユーザー定義の辞書です。この辞書のためにUIはJSONビューアをレンダリングし、_kind キーを持つ辞書に特別な処理を施します。これは、モデルベンダーが非常に異なるシリアライゼーションフォーマットを持っているため、ここで柔軟である必要があるからです。
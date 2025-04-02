---
title: WBTraceTree
menu:
  reference:
    identifier: ja-ref-python-data-types-wbtracetree
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/trace_tree.py#L80-L119 >}}

trace ツリー データ用の Media オブジェクト。

```python
WBTraceTree(
    root_span: Span,
    model_dict: typing.Optional[dict] = None
)
```

| Args |  |
| :--- | :--- |
| root_span (Span): trace ツリーのルート スパン。model_dict (dict, optional): モデル ダンプを含む辞書。注: model_dict は完全に ユーザー 定義の辞書です。UI はこの辞書の JSON ビューアをレンダリングし、`_kind` キーを持つ辞書を特別に扱います。これは、モデル ベンダーが非常に異なるシリアル化形式を持っているため、ここで柔軟に対応する必要があるためです。 |

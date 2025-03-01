---
title: WBTraceTree
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/2678738e59629208ad4770e3d36300a272147c05/wandb/sdk/data_types/trace_tree.py#L80-L119 >}}

Media object for trace tree data.

```python
WBTraceTree(
    root_span: Span,
    model_dict: typing.Optional[dict] = None
)
```

| Args |  |
| :--- | :--- |
|  root_span (Span): The root span of the trace tree. model_dict (dict, optional): A dictionary containing the model dump. NOTE: model_dict is a completely-user-defined dict. The UI will render a JSON viewer for this dict, giving special treatment to dictionaries with a `_kind` key. This is because model vendors have such different serialization formats that we need to be flexible here. |

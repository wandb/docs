---
title: エージェント()
data_type_classification: function
menu:
  reference:
    identifier: ja-ref-python-sdk-functions-agent
object_type: python_sdk_actions
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb_agent.py >}}




### <kbd>function</kbd> `agent`

```python
agent(
    sweep_id: str,
    function: Optional[Callable] = None,
    entity: Optional[str] = None,
    project: Optional[str] = None,
    count: Optional[int] = None
) → None
```

1 つ以上の sweep agent を起動します。

sweep agent は `sweep_id` を使って、自分がどの sweep に属しているか、どの関数を実行するか、そして（オプションで）起動する sweep agent の数を判断します。



**Args:**

 - `sweep_id`:  sweep の一意な識別子。sweep ID は W&B CLI または Python SDK によって生成されます。 
 - `function`:  sweep config で指定された "program" の代わりに呼び出す関数。 
 - `entity`:  sweep で作成された W&B runs を送信したい username または team name。指定した entity が既に存在していることを確認してください。entity を指定しない場合、run はデフォルトの entity（通常はあなたの username）に送信されます。 
 - `project`:  sweep から作成された W&B runs の送信先となる project の名前。project が指定されていない場合、run は "Uncategorized" というラベルの project に送信されます。 
 - `count`:  試す sweep config のトライアル数。
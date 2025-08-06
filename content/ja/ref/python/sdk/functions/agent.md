---
title: agent()
object_type: python_sdk_actions
data_type_classification: function
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

sweep agent は `sweep_id` を使って、どの sweep に属するか、どの関数を実行するか、そして（オプションで）何台の agent を起動するかを判断します。



**引数:**
 
 - `sweep_id`:  sweep の一意な識別子です。W&B CLI または Python SDK で sweep ID が生成されます。
 - `function`:  sweep の設定ファイルで指定された "program" の代わりに実行する関数。
 - `entity`:  sweep で作成される W&B run を送信したいユーザー名またはチーム名。指定した entity がすでに存在している必要があります。entity を指定しない場合、run はデフォルトの entity（通常はあなたのユーザー名）に送られます。
 - `project`:  sweep から作成された W&B run を送信する project の名前。project を指定しない場合は「Uncategorized」と表示される project に送信されます。
 - `count`:  sweep の設定で実行するトライアル数。
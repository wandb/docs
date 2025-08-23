---
title: 'agent()

  '
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

1つまたは複数の sweep agent を起動します。

sweep agent は `sweep_id` を使って、どの sweep に属しているか、実行する関数は何か、（必要に応じて）いくつの agent を実行するかを把握します。

**引数:**
 
 - `sweep_id`:  sweep の一意な識別子。W&B CLI または Python SDK で sweep ID が生成されます。
 - `function`:  sweep 設定ファイルの "program" の代わりに実行する関数。
 - `entity`:  sweep で作成される W&B run を送信したいユーザー名またはチーム名。指定した entity が既に存在することを確認してください。entity を指定しない場合、run はデフォルトの entity（通常は自分のユーザー名）に送信されます。
 - `project`:  sweep から生成された W&B run を送るプロジェクト名。project を指定しない場合、run は「Uncategorized」とラベル付けされた project に送信されます。
 - `count`:  sweep 設定で試行するトライアルの数。
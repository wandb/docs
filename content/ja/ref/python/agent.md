---
title: agent
menu:
  reference:
    identifier: ja-ref-python-agent
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/wandb_agent.py#L532-L576 >}}

1 つ以上の sweep agent を起動します。

```python
agent(
    sweep_id: str,
    function: Optional[Callable] = None,
    entity: Optional[str] = None,
    project: Optional[str] = None,
    count: Optional[int] = None
) -> None
```

sweep agent は、どの sweep の一部であるか、どの関数を実行するか、(オプションで) 実行する agent の数を `sweep_id` を使用して認識します。

| Args |  |
| :--- | :--- |
|  `sweep_id` |  sweep の一意の識別子。 sweep ID は、W&B CLI または Python SDK によって生成されます。 |
|  `function` |  sweep config で指定された「program」の代わりに使用する関数。 |
|  `entity` |  sweep によって作成された W&B run の送信先となる、 ユーザー 名または Team 名。指定する entity がすでに存在することを確認してください。 entity を指定しない場合、run はデフォルトの entity (通常は ユーザー 名) に送信されます。 |
|  `project` |  sweep から作成された W&B run の送信先となる Project の名前。 Project が指定されていない場合、run は「Uncategorized」というラベルの Project に送信されます。 |
|  `count` |  試行する sweep config トライアル の数。 |

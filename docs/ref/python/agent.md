
# エージェント

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/wandb_agent.py#L534-L579' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

1つ以上のsweepエージェントを開始します。

```python
agent(
    sweep_id: str,
    function: Optional[Callable] = None,
    entity: Optional[str] = None,
    project: Optional[str] = None,
    count: Optional[int] = None
) -> None
```

sweepエージェントは、`sweep_id`を使用して、どのsweepに属しているか、どの関数を実行するか、および（任意で）いくつのエージェントを実行するかを判断します。

| 引数 |  |
| :--- | :--- |
|  `sweep_id` |  sweepの一意の識別子。sweep IDはW&B CLIまたはPython SDKによって生成されます。 |
|  `function` |  sweep設定で指定された「プログラム」の代わりに呼び出される関数。 |
|  `entity` |  sweepによって作成されたW&B runsを送信したいユーザー名またはチーム名。指定したentityが既に存在することを確認してください。entityを指定しない場合、runはデフォルトのentity（通常はユーザー名）に送信されます。 |
|  `project` |  sweepから作成されたW&B runsが送信されるプロジェクトの名前。プロジェクトが指定されていない場合、runは「Uncategorized」とラベル付けされたプロジェクトに送信されます。 |
|  `count` |  試行するsweep設定トライアルの数。 |
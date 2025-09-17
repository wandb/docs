---
title: LaunchAgent
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L164-L924 >}}

Launch エージェント クラス。指定された run キューをポーリングし、wandb launch のために run を起動します。

```python
LaunchAgent(
    api: Api,
    config: Dict[str, Any]
)
```

| 引数 |  |
| :--- | :--- |
|  `api` |  バックエンドにリクエストするために使用する Api オブジェクト。 |
|  `config` |  エージェントの Config 辞書。 |

| 属性 |  |
| :--- | :--- |
|  `num_running_jobs` |  スケジューラを除いたジョブ数を返します。 |
|  `num_running_schedulers` |  スケジューラの数のみを返します。 |
|  `thread_ids` |  エージェントの実行中スレッド ID のキー リストを返します。 |

## メソッド

### `check_sweep_state`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L786-L803)

```python
check_sweep_state(
    launch_spec, api
)
```

sweep のために run を起動する前に、その状態を確認します。

### `fail_run_queue_item`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L295-L304)

```python
fail_run_queue_item(
    run_queue_item_id, message, phase, files=None
)
```

### `finish_thread_id`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L416-L509)

```python
finish_thread_id(
    thread_id, exception=None
)
```

一時的にこのジョブをリストから外します。

### `get_job_and_queue`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L908-L915)

```python
get_job_and_queue()
```

### `initialized`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L190-L193)

```python
@classmethod
initialized() -> bool
```

エージェントが初期化済みかどうかを返します。

### `loop`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L572-L653)

```python
loop()
```

ジョブをポーリングして実行する無限ループです。

| 例外 |  |
| :--- | :--- |
|  `KeyboardInterrupt` |  エージェントの停止が要求された場合。 |

### `name`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L180-L188)

```python
@classmethod
name() -> str
```

エージェント名を返します。

### `pop_from_queue`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L340-L363)

```python
pop_from_queue(
    queue
)
```

ジョブとして実行するために run キューからアイテムを 1 つ取り出します。

| 引数 |  |
| :--- | :--- |
|  `queue` |  取り出し元のキュー。 |

| 戻り値 |  |
| :--- | :--- |
|  キューから取り出したアイテム。 |

| 例外 |  |
| :--- | :--- |
|  `Exception` |  キューからの取り出し時にエラーが発生した場合。 |

### `print_status`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L365-L381)

```python
print_status() -> None
```

エージェントの現在のステータスを出力します。

### `run_job`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L511-L541)

```python
run_job(
    job, queue, file_saver
)
```

project をセットアップし、ジョブを実行します。

| 引数 |  |
| :--- | :--- |
|  `job` |  実行するジョブ。 |

### `task_run_job`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L656-L688)

```python
task_run_job(
    launch_spec, job, default_config, api, job_tracker
)
```

### `update_status`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L383-L394)

```python
update_status(
    status
)
```

エージェントのステータスを更新します。

| 引数 |  |
| :--- | :--- |
|  `status` |  エージェントを更新するステータス。 |
---
title: ローンンチエージェント
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L164-L924 >}}

Launch agent クラスは、指定された run キューをポーリングし、wandb launch のために run を起動します。

```python
LaunchAgent(
    api: Api,
    config: Dict[str, Any]
)
```

| 引数 |  |
| :--- | :--- |
|  `api` |  バックエンドへのリクエストに使用する Api オブジェクト。 |
|  `config` |  agent 用の設定辞書。 |

| 属性 |  |
| :--- | :--- |
|  `num_running_jobs` |  スケジューラーを除く実行中のジョブ数を返します。 |
|  `num_running_schedulers` |  スケジューラーの数のみを返します。 |
|  `thread_ids` |  agent で実行中のスレッド ID キーのリストを返します。 |

## メソッド

### `check_sweep_state`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L786-L803)

```python
check_sweep_state(
    launch_spec, api
)
```

sweep 用の run を起動する前に、sweep の状態を確認します。

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

現在のリストからそのジョブを削除します。

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

agent が初期化されているかどうかを返します。

### `loop`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L572-L653)

```python
loop()
```

ジョブをポーリングして実行するために無限ループします。

| 例外 |  |
| :--- | :--- |
|  `KeyboardInterrupt` |  agent の停止が要求された場合に発生します。 |

### `name`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L180-L188)

```python
@classmethod
name() -> str
```

agent の名前を返します。

### `pop_from_queue`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L340-L363)

```python
pop_from_queue(
    queue
)
```

runqueue からアイテムを取り出してジョブとして実行します。

| 引数 |  |
| :--- | :--- |
|  `queue` |  取り出す対象のキュー。 |

| 戻り値 |  |
| :--- | :--- |
|  キューから取り出されたアイテム。 |

| 例外 |  |
| :--- | :--- |
|  `Exception` |  キューから取り出す際にエラーが発生した場合にスローされます。 |

### `print_status`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L365-L381)

```python
print_status() -> None
```

agent の現在のステータスを表示します。

### `run_job`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L511-L541)

```python
run_job(
    job, queue, file_saver
)
```

Project をセットアップし、ジョブを実行します。

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

agent のステータスを更新します。

| 引数 |  |
| :--- | :--- |
|  `status` |  agent を更新するステータス。 |
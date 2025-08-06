---
title: ローンチエージェント
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L164-L924 >}}

Launch エージェントクラスは、指定された run キューから任意の run をポーリングし、wandb launch のための run を起動します。

```python
LaunchAgent(
    api: Api,
    config: Dict[str, Any]
)
```

| 引数 |  |
| :--- | :--- |
|  `api` |  バックエンドへのリクエストを行うために使用する Api オブジェクト。 |
|  `config` |  エージェント用の設定辞書。 |

| 属性 |  |
| :--- | :--- |
|  `num_running_jobs` |  スケジューラーを除いた現在実行中のジョブ数を返します。 |
|  `num_running_schedulers` |  スケジューラーのみの実行中数を返します。 |
|  `thread_ids` |  エージェントが実行しているスレッド ID のリスト（キー）を返します。 |

## メソッド

### `check_sweep_state`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L786-L803)

```python
check_sweep_state(
    launch_spec, api
)
```

sweep 用の run を起動する前に、その sweep の状態を確認します。

### `fail_run_queue_item`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L295-L304)

```python
fail_run_queue_item(
    run_queue_item_id, message, phase, files=None
)
```

### `finish_thread_id`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L416-L509)

```python
finish_thread_id(
    thread_id, exception=None
)
```

現時点でリストからジョブを削除します。

### `get_job_and_queue`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L908-L915)

```python
get_job_and_queue()
```

### `initialized`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L190-L193)

```python
@classmethod
initialized() -> bool
```

エージェントが初期化されているかどうかを返します。

### `loop`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L572-L653)

```python
loop()
```

無限ループでジョブをポーリングし、実行します。

| 発生する例外 |  |
| :--- | :--- |
|  `KeyboardInterrupt` |  エージェントが停止リクエストされた場合。 |

### `name`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L180-L188)

```python
@classmethod
name() -> str
```

エージェントの名前を返します。

### `pop_from_queue`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L340-L363)

```python
pop_from_queue(
    queue
)
```

runキューから 1 件を取り出してジョブとして実行します。

| 引数 |  |
| :--- | :--- |
|  `queue` |  取り出し元となるキュー。 |

| 戻り値 |  |
| :--- | :--- |
|  キューから取り出したアイテム。 |

| 発生する例外 |  |
| :--- | :--- |
|  `Exception` |  キューからの取り出し時にエラーが発生した場合。 |

### `print_status`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L365-L381)

```python
print_status() -> None
```

エージェントの現在のステータスを表示します。

### `run_job`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L511-L541)

```python
run_job(
    job, queue, file_saver
)
```

Project の準備を行いジョブを実行します。

| 引数 |  |
| :--- | :--- |
|  `job` |  実行するジョブ。 |

### `task_run_job`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L656-L688)

```python
task_run_job(
    launch_spec, job, default_config, api, job_tracker
)
```

### `update_status`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L383-L394)

```python
update_status(
    status
)
```

エージェントの状態を更新します。

| 引数 |  |
| :--- | :--- |
|  `status` |  エージェントの状態を更新するステータス。 |
---
title: LaunchAgent
menu:
  reference:
    identifier: ja-ref-python-launch-library-launchagent
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/launch/agent/agent.py#L164-L924 >}}

Launch エージェントクラスは、指定されたランキューをポーリングし、wandb Launch のために runs をローンチします。

```python
LaunchAgent(
    api: Api,
    config: Dict[str, Any]
)
```

| 引数 |  |
| :--- | :--- |
|  `api` |  バックエンドへのリクエストを行うために使用する Api オブジェクト。 |
|  `config` |  エージェントの設定辞書。 |

| 属性 |  |
| :--- | :--- |
|  `num_running_jobs` |  スケジューラを除くジョブの数を返します。 |
|  `num_running_schedulers` |  スケジューラの数だけを返します。 |
|  `thread_ids` |  エージェントに対してスレッド ID を実行しているキーのリストを返します。 |

## メソッド

### `check_sweep_state`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/launch/agent/agent.py#L786-L803)

```python
check_sweep_state(
    launch_spec, api
)
```

sweep のために run をローンチする前に sweep の状態を確認します。

### `fail_run_queue_item`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/launch/agent/agent.py#L295-L304)

```python
fail_run_queue_item(
    run_queue_item_id, message, phase, files=None
)
```

### `finish_thread_id`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/launch/agent/agent.py#L416-L509)

```python
finish_thread_id(
    thread_id, exception=None
)
```

しばらくの間、ジョブをリストから削除します。

### `get_job_and_queue`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/launch/agent/agent.py#L908-L915)

```python
get_job_and_queue()
```

### `initialized`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/launch/agent/agent.py#L190-L193)

```python
@classmethod
initialized() -> bool
```

エージェントが初期化されているかを返します。

### `loop`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/launch/agent/agent.py#L572-L653)

```python
loop()
```

ジョブをポーリングしてそれを実行するために無限ループします。

| 例外 |  |
| :--- | :--- |
|  `KeyboardInterrupt` |  エージェントが停止を要求された場合。 |

### `name`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/launch/agent/agent.py#L180-L188)

```python
@classmethod
name() -> str
```

エージェントの名前を返します。

### `pop_from_queue`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/launch/agent/agent.py#L340-L363)

```python
pop_from_queue(
    queue
)
```

ジョブとして実行するために run キューからアイテムを取り出します。

| 引数 |  |
| :--- | :--- |
|  `queue` |  取り出すためのキュー。 |

| 戻り値 |  |
| :--- | :--- |
|  キューから取り出されたアイテム。 |

| 例外 |  |
| :--- | :--- |
|  `Exception` |  キューからの取り出しにエラーがある場合。 |

### `print_status`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/launch/agent/agent.py#L365-L381)

```python
print_status() -> None
```

エージェントの現在のステータスを表示します。

### `run_job`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/launch/agent/agent.py#L511-L541)

```python
run_job(
    job, queue, file_saver
)
```

プロジェクトをセットアップし、ジョブを実行します。

| 引数 |  |
| :--- | :--- |
|  `job` |  実行するジョブ。 |

### `task_run_job`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/launch/agent/agent.py#L656-L688)

```python
task_run_job(
    launch_spec, job, default_config, api, job_tracker
)
```

### `update_status`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/launch/agent/agent.py#L383-L394)

```python
update_status(
    status
)
```

エージェントのステータスを更新します。

| 引数 |  |
| :--- | :--- |
|  `status` |  エージェントを更新するステータス。 |
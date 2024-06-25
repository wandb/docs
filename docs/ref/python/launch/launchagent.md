
# LaunchAgent

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/launch/agent/agent.py#L164-L918' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

Launch agent クラスは指定された run キューをポーリングし、wandb launch のための runs を開始します。

```python
LaunchAgent(
    api: Api,
    config: Dict[str, Any]
)
```

| 引数 |  |
| :--- | :--- |
|  `api` |  バックエンドにリクエストを送るための Api オブジェクト。 |
|  `config` |  エージェントのための設定辞書。 |

| 属性 |  |
| :--- | :--- |
|  `num_running_jobs` | スケジューラを除いた現在実行中のジョブ数を返します。 |
|  `num_running_schedulers` | スケジューラの数のみを返します。 |
|  `thread_ids` | エージェントに対して実行中のスレッドIDのリストを返します。 |

## メソッド

### `check_sweep_state`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/launch/agent/agent.py#L780-L797)

```python
check_sweep_state(
    launch_spec, api
)
```

sweep の実行前に sweep の状態を確認します。

### `fail_run_queue_item`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/launch/agent/agent.py#L295-L304)

```python
fail_run_queue_item(
    run_queue_item_id, message, phase, files=None
)
```

### `finish_thread_id`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/launch/agent/agent.py#L414-L507)

```python
finish_thread_id(
    thread_id, exception=None
)
```

現時点でリストからジョブを削除します。

### `get_job_and_queue`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/launch/agent/agent.py#L902-L909)

```python
get_job_and_queue()
```

### `initialized`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/launch/agent/agent.py#L190-L193)

```python
@classmethod
initialized() -> bool
```

エージェントが初期化されているかどうか返します。

### `loop`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/launch/agent/agent.py#L570-L651)

```python
loop()
```

ジョブをポーリングして実行し続けます。

| 発生する例外 |  |
| :--- | :--- |
|  `KeyboardInterrupt` |  エージェントを停止する要求があった場合。 |

### `name`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/launch/agent/agent.py#L180-L188)

```python
@classmethod
name() -> str
```

エージェントの名前を返します。

### `pop_from_queue`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/launch/agent/agent.py#L338-L361)

```python
pop_from_queue(
    queue
)
```

run キューからアイテムをポップしてジョブとして実行します。

| 引数 |  |
| :--- | :--- |
|  `queue` |  ポップするキュー。 |

| 返り値 |  |
| :--- | :--- |
|  キューからポップされたアイテム。 |

| 発生する例外 |  |
| :--- | :--- |
|  `Exception` |  キューからポップする際にエラーが発生した場合。 |

### `print_status`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/launch/agent/agent.py#L363-L379)

```python
print_status() -> None
```

エージェントの現在のステータスを表示します。

### `run_job`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/launch/agent/agent.py#L509-L539)

```python
run_job(
    job, queue, file_saver
)
```

プロジェクトをセットアップしてジョブを実行します。

| 引数 |  |
| :--- | :--- |
|  `job` |  実行するジョブ。 |

### `task_run_job`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/launch/agent/agent.py#L654-L686)

```python
task_run_job(
    launch_spec, job, default_config, api, job_tracker
)
```

### `update_status`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/launch/agent/agent.py#L381-L392)

```python
update_status(
    status
)
```

エージェントのステータスを更新します。

| 引数 |  |
| :--- | :--- |
|  `status` |  エージェントの新しいステータス。 |
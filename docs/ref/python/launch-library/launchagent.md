# LaunchAgent

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/launch/agent/agent.py#L164-L924' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

주어진 run 대기열을 폴링하고 W&B Launch를 위해 run을 시작하는 Launch agent 클래스입니다.

```python
LaunchAgent(
    api: Api,
    config: Dict[str, Any]
)
```

| 인수 |  |
| :--- | :--- |
|  `api` |  백엔드로 요청을 보내는 데 사용할 Api 오브젝트. |
|  `config` |  에이전트를 위한 설정 사전. |

| 속성 |  |
| :--- | :--- |
|  `num_running_jobs` |  스케줄러를 제외한 실행 중인 작업의 수를 반환합니다. |
|  `num_running_schedulers` |  스케줄러의 수만 반환합니다. |
|  `thread_ids` |  에이전트의 실행 중인 스레드 ID 목록을 반환합니다. |

## 메소드

### `check_sweep_state`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/launch/agent/agent.py#L786-L803)

```python
check_sweep_state(
    launch_spec, api
)
```

run을 시작하기 전에 sweep의 상태를 확인합니다.

### `fail_run_queue_item`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/launch/agent/agent.py#L295-L304)

```python
fail_run_queue_item(
    run_queue_item_id, message, phase, files=None
)
```

### `finish_thread_id`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/launch/agent/agent.py#L416-L509)

```python
finish_thread_id(
    thread_id, exception=None
)
```

지금은 목록에서 작업을 제거합니다.

### `get_job_and_queue`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/launch/agent/agent.py#L908-L915)

```python
get_job_and_queue()
```

### `initialized`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/launch/agent/agent.py#L190-L193)

```python
@classmethod
initialized() -> bool
```

에이전트가 초기화되었는지 여부를 반환합니다.

### `loop`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/launch/agent/agent.py#L572-L653)

```python
loop()
```

작업을 폴링하고 실행하기 위해 무한히 루프를 반복합니다.

| 발생 |  |
| :--- | :--- |
|  `KeyboardInterrupt` |  에이전트가 정지를 요청받은 경우. |

### `name`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/launch/agent/agent.py#L180-L188)

```python
@classmethod
name() -> str
```

에이전트의 이름을 반환합니다.

### `pop_from_queue`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/launch/agent/agent.py#L340-L363)

```python
pop_from_queue(
    queue
)
```

run 대기열에서 항목을 추출하여 작업으로 실행합니다.

| 인수 |  |
| :--- | :--- |
|  `queue` |  추출할 대기열. |

| 반환 |  |
| :--- | :--- |
|  대기열에서 추출된 항목. |

| 발생 |  |
| :--- | :--- |
|  `Exception` |  대기열에서 추출하는 동안 오류가 발생한 경우. |

### `print_status`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/launch/agent/agent.py#L365-L381)

```python
print_status() -> None
```

에이전트의 현재 상태를 출력합니다.

### `run_job`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/launch/agent/agent.py#L511-L541)

```python
run_job(
    job, queue, file_saver
)
```

프로젝트를 설정하고 작업을 실행합니다.

| 인수 |  |
| :--- | :--- |
|  `job` |  실행할 작업. |

### `task_run_job`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/launch/agent/agent.py#L656-L688)

```python
task_run_job(
    launch_spec, job, default_config, api, job_tracker
)
```

### `update_status`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/launch/agent/agent.py#L383-L394)

```python
update_status(
    status
)
```

에이전트의 상태를 업데이트합니다.

| 인수 |  |
| :--- | :--- |
|  `status` |  에이전트를 업데이트할 상태. |
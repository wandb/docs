---
title: LaunchAgent
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L164-L924 >}}

Launch 에이전트 클래스는 주어진 run 큐를 폴링하여 wandb launch를 위한 run 들을 실행합니다.

```python
LaunchAgent(
    api: Api,
    config: Dict[str, Any]
)
```

| 인수 |  |
| :--- | :--- |
|  `api` |  백엔드에 요청할 때 사용할 Api 오브젝트입니다. |
|  `config` |  에이전트의 설정 사전(config dictionary)입니다. |

| 속성 |  |
| :--- | :--- |
|  `num_running_jobs` |  스케쥴러를 제외한 실행 중인 잡의 개수를 반환합니다. |
|  `num_running_schedulers` |  실행 중인 스케쥴러의 개수만 반환합니다. |
|  `thread_ids` |  에이전트가 실행 중인 쓰레드 아이디 리스트(키)를 반환합니다. |

## 메소드

### `check_sweep_state`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L786-L803)

```python
check_sweep_state(
    launch_spec, api
)
```

스윕을 시작하기 전에 해당 스윕 상태를 확인합니다.

### `fail_run_queue_item`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L295-L304)

```python
fail_run_queue_item(
    run_queue_item_id, message, phase, files=None
)
```

### `finish_thread_id`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L416-L509)

```python
finish_thread_id(
    thread_id, exception=None
)
```

해당 잡을 현재 관리 리스트에서 제거합니다.

### `get_job_and_queue`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L908-L915)

```python
get_job_and_queue()
```

### `initialized`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L190-L193)

```python
@classmethod
initialized() -> bool
```

에이전트가 초기화되었는지 여부를 반환합니다.

### `loop`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L572-L653)

```python
loop()
```

잡을 폴링하여 지속적으로 run 을 실행하는 무한 루프입니다.

| 발생 예외 |  |
| :--- | :--- |
|  `KeyboardInterrupt` |  에이전트에게 중지 요청이 들어오면 발생합니다. |

### `name`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L180-L188)

```python
@classmethod
name() -> str
```

에이전트의 이름을 반환합니다.

### `pop_from_queue`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L340-L363)

```python
pop_from_queue(
    queue
)
```

runqueue에서 하나를 pop하여 잡으로 실행합니다.

| 인수 |  |
| :--- | :--- |
|  `queue` |  pop 할 대상 큐입니다. |

| 반환값 |  |
| :--- | :--- |
|  큐에서 pop된 항목입니다. |

| 발생 예외 |  |
| :--- | :--- |
|  `Exception` |  큐에서 pop할 때 오류가 있으면 발생합니다. |

### `print_status`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L365-L381)

```python
print_status() -> None
```

에이전트의 현재 상태를 출력합니다.

### `run_job`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L511-L541)

```python
run_job(
    job, queue, file_saver
)
```

프로젝트를 세팅하고 잡을 실행합니다.

| 인수 |  |
| :--- | :--- |
|  `job` |  실행할 잡입니다. |

### `task_run_job`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L656-L688)

```python
task_run_job(
    launch_spec, job, default_config, api, job_tracker
)
```

### `update_status`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/sdk/launch/agent/agent.py#L383-L394)

```python
update_status(
    status
)
```

에이전트의 상태를 업데이트합니다.

| 인수 |  |
| :--- | :--- |
|  `status` |  에이전트의 상태로 업데이트할 값입니다. |
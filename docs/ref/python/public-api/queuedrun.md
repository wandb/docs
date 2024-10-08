# QueuedRun

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/jobs.py#L220-L423' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHub에서 소스 보기</a></button></p>

엔티티 및 프로젝트와 연결된 한 개의 대기열에 있는 run입니다. `run = queued_run.wait_until_running()` 또는 `run = queued_run.wait_until_finished()`를 호출하여 run에 엑세스하세요.

```python
QueuedRun(
    client, entity, project, queue_name, run_queue_item_id,
    project_queue=LAUNCH_DEFAULT_PROJECT, priority=None
)
```

| 속성 |  |
| :--- | :--- |

## 메소드

### `delete`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/jobs.py#L344-L393)

```python
delete(
    delete_artifacts=(False)
)
```

wandb 백엔드에서 주어진 대기열에 있는 run을 삭제합니다.

### `wait_until_finished`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/jobs.py#L334-L342)

```python
wait_until_finished()
```

### `wait_until_running`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/jobs.py#L395-L420)

```python
wait_until_running()
```
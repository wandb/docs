
# QueuedRun

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/jobs.py#L202-L405' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHub에서 소스 보기</a></button></p>

엔티티와 프로젝트에 연결된 단일 대기중인 실행입니다. `run = queued_run.wait_until_running()` 또는 `run = queued_run.wait_until_finished()`를 호출하여 실행에 엑세스하세요.

```python
QueuedRun(
    client, entity, project, queue_name, run_queue_item_id,
    project_queue=LAUNCH_DEFAULT_PROJECT, priority=None
)
```

| 속성 |  |
| :--- | :--- |

## 메서드

### `delete`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/jobs.py#L326-L375)

```python
delete(
    delete_artifacts=(False)
)
```

wandb 백엔드에서 주어진 대기중인 실행을 삭제합니다.

### `wait_until_finished`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/jobs.py#L316-L324)

```python
wait_until_finished()
```

### `wait_until_running`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/jobs.py#L377-L402)

```python
wait_until_running()
```
# QueuedRun

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/jobs.py#L215-L418' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

エンティティとプロジェクトに関連する単一のキューに入れられた run。 `run = queued_run.wait_until_running()` また、 `run = queued_run.wait_until_finished()` を呼び出して run にアクセスします。

```python
QueuedRun(
    client, entity, project, queue_name, run_queue_item_id,
    project_queue=LAUNCH_DEFAULT_PROJECT, priority=None
)
```

| 属性 |  |
| :--- | :--- |

## メソッド

### `delete`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/jobs.py#L339-L388)

```python
delete(
    delete_artifacts=(False)
)
```

指定されたキューに入れられた run を wandb バックエンドから削除します。

### `wait_until_finished`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/jobs.py#L329-L337)

```python
wait_until_finished()
```

### `wait_until_running`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/jobs.py#L390-L415)

```python
wait_until_running()
```
# Job

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/jobs.py#L36-L217' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>소스 보기 GitHub에서</a></button></p>


```python
Job(
    api: "Api",
    name,
    path: Optional[str] = None
) -> None
```

| 속성 |  |
| :--- | :--- |

## 메소드

### `call`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/jobs.py#L173-L217)

```python
call(
    config, project=None, entity=None, queue=None, resource="local-container",
    resource_args=None, template_variables=None, project_queue=None, priority=None
)
```

### `set_entrypoint`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/jobs.py#L170-L171)

```python
set_entrypoint(
    entrypoint: List[str]
)
```
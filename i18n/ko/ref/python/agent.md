
# 에이전트

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/wandb_agent.py#L534-L579' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHub에서 소스 보기</a></button></p>

하나 이상의 스윕 에이전트를 시작합니다.

```python
agent(
    sweep_id: str,
    function: Optional[Callable] = None,
    entity: Optional[str] = None,
    project: Optional[str] = None,
    count: Optional[int] = None
) -> None
```

스윕 에이전트는 `sweep_id`를 사용하여 어떤 스윕의 일부인지, 실행할 함수가 무엇인지, 그리고 (선택적으로) 몇 명의 에이전트를 실행할지를 알아냅니다.

| 인수 |  |
| :--- | :--- |
|  `sweep_id` |  스윕의 고유 식별자입니다. 스윕 ID는 W&B CLI 또는 Python SDK에 의해 생성됩니다. |
|  `function` |  스윕 구성에서 지정된 "프로그램" 대신 호출할 함수입니다. |
|  `entity` |  스윕에 의해 생성된 W&B 실행을 보낼 사용자 이름 또는 팀 이름입니다. 지정하는 엔티티가 이미 존재하는지 확인하세요. 엔티티를 지정하지 않으면 실행은 보통 사용자 이름인 기본 엔티티로 보내집니다. |
|  `project` |  스윕에서 생성된 W&B 실행이 보내지는 프로젝트의 이름입니다. 프로젝트를 지정하지 않으면 실행은 "Uncategorized"로 표시된 프로젝트로 보내집니다. |
|  `count` |  시도할 스윕 구성 시행의 수입니다. |
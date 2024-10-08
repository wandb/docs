# sweep

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_sweep.py#L34-L92' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

하이퍼파라미터 탐색을 초기화합니다.

```python
sweep(
    sweep: Union[dict, Callable],
    entity: Optional[str] = None,
    project: Optional[str] = None,
    prior_runs: Optional[List[str]] = None
) -> str
```

기계학습 모델의 비용 함수 최적화를 위한 하이퍼파라미터를 다양한 조합으로 테스트하여 탐색합니다.

고유 식별자 `sweep_id`를 기록해 두세요. 나중 단계에서 `sweep_id`를 스윕 에이전트에 제공해야 합니다.

| 인수 |  |
| :--- | :--- |
|  `sweep` |  하이퍼파라미터 탐색의 설정입니다. (또는 설정 생성기). 스윕을 정의하는 방법에 대한 정보는 [스윕 구성 구조](/guides/sweeps/define-sweep-configuration)를 참조하세요. 콜러블을 제공할 경우, 해당 콜러블이 인수를 받지 않으며 W&B 스윕 구성 사양에 맞는 사전을 반환해야 합니다. |
|  `entity` |  스윕에 의해 생성된 W&B runs를 보낼 사용자 이름 또는 팀 이름입니다. 지정한 엔티티가 이미 존재하는지 확인하세요. 엔티티를 지정하지 않으면 run이 기본 엔티티(일반적으로 사용자 이름)로 전송됩니다. |
|  `project` |  스윕에서 생성된 W&B runs가 전송될 프로젝트 이름입니다. 프로젝트가 지정되지 않으면 run은 'Uncategorized'로 라벨링된 프로젝트로 전송됩니다. |
|  `prior_runs` |  이 스윕에 추가할 기존 run의 run ID입니다. |

| 반환 값 |  |
| :--- | :--- |
|  `sweep_id` |  str. 스윕의 고유 식별자입니다. |
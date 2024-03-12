
# 스윕

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/wandb_sweep.py#L31-L87' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHub에서 소스 보기</a></button></p>


하이퍼파라미터 탐색을 초기화합니다.

```python
sweep(
    sweep: Union[dict, Callable],
    entity: Optional[str] = None,
    project: Optional[str] = None
) -> str
```

다양한 조합을 테스트하여 기계학습 모델의 비용 함수를 최적화하는 하이퍼파라미터를 찾습니다.

반환되는 고유 식별자, `sweep_id`를 기록하세요.
나중에 스윕 에이전트에 `sweep_id`를 제공합니다.

| 인수 |  |
| :--- | :--- |
|  `sweep` |  하이퍼파라미터 탐색의 구성입니다. (또는 구성 생성기). 스윕을 정의하는 방법에 대한 정보는 [스윕 구성 구조](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration)에서 확인하세요. 호출 가능한 객체를 제공하는 경우, 이 객체가 인수를 취하지 않고 W&B 스윕 구성 사양에 맞는 사전을 반환한다는 것을 확인해야 합니다. |
|  `entity` |  스윕에 의해 생성된 W&B run을 보내고자 하는 사용자명 또는 팀명입니다. 지정한 엔터티가 이미 존재하는지 확인하세요. 엔터티를 지정하지 않으면, run은 보통 사용자명인 기본 엔터티로 전송됩니다. |
|  `project` |  스윕에서 생성된 W&B run이 전송되는 프로젝트의 이름입니다. 프로젝트가 지정되지 않은 경우, run은 'Uncategorized'라는 레이블이 붙은 프로젝트로 전송됩니다. |

| 반환값 |  |
| :--- | :--- |
|  `sweep_id` |  str. 스윕에 대한 고유 식별자입니다. |
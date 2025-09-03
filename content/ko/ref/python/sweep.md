---
title: sweep
menu:
  reference:
    identifier: ko-ref-python-sweep
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_sweep.py#L34-L92 >}}

하이퍼파라미터 스윕을 초기화합니다.

```python
sweep(
    sweep: Union[dict, Callable],
    entity: Optional[str] = None,
    project: Optional[str] = None,
    prior_runs: Optional[List[str]] = None
) -> str
```

다양한 조합을 테스트하여 기계 학습 모델의 비용 함수를 최적화하는 하이퍼파라미터를 검색합니다.

반환되는 고유 식별자인 `sweep_id` 를 기록해 두십시오.
나중에 스윕 에이전트에 `sweep_id` 를 제공합니다.

| 인수 |  |
| :--- | :--- |
|  `sweep` | 하이퍼파라미터 검색의 구성입니다 (또는 구성 생성기). 스윕 정의 방법에 대한 자세한 내용은 [스윕 구성 구조](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration) 를 참조하십시오. 호출 가능한 항목을 제공하는 경우 호출 가능한 항목이 인수를 사용하지 않고 W&B 스윕 구성 사양을 준수하는 사전을 반환하는지 확인하십시오. |
|  `entity` | 스윕에 의해 생성된 W&B run을 보낼 사용자 이름 또는 팀 이름입니다. 지정한 엔터티가 이미 존재하는지 확인하십시오. 엔터티를 지정하지 않으면 run은 기본 엔터티 (일반적으로 사용자 이름) 로 전송됩니다. |
|  `project` | 스윕에서 생성된 W&B run이 전송되는 프로젝트의 이름입니다. 프로젝트를 지정하지 않으면 run은 'Uncategorized' 라는 프로젝트로 전송됩니다. |
|  `prior_runs` | 이 스윕에 추가할 기존 run의 run ID입니다. |

| 반환 |  |
| :--- | :--- |
|  `sweep_id` |  str. 스윕에 대한 고유 식별자입니다. |

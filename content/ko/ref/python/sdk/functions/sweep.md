---
title: sweep()
data_type_classification: function
menu:
  reference:
    identifier: ko-ref-python-sdk-functions-sweep
object_type: python_sdk_actions
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_sweep.py >}}




### <kbd>function</kbd> `sweep`

```python
sweep(
    sweep: Union[dict, Callable],
    entity: Optional[str] = None,
    project: Optional[str] = None,
    prior_runs: Optional[List[str]] = None
) → str
```

하이퍼파라미터 탐색(hyperparameter sweep)을 초기화합니다.

기계학습 모델의 비용 함수(cost function)를 최적화하는 하이퍼파라미터를 찾기 위해 다양한 조합을 실험합니다.

반환되는 고유 식별자인 `sweep_id`를 꼭 기록해두세요. 나중 단계에서 이 `sweep_id`를 sweep agent에 전달해야 합니다.

스윕 정의 방법에 대한 자세한 내용은 [Sweep configuration structure](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration) 문서를 참고하세요.



**인수:**
 
 - `sweep`:  하이퍼파라미터 탐색의 설정(또는 설정을 생성하는 함수).  callable을 제공하는 경우, 해당 callable이 인수를 받지 않아야 하며, W&B 스윕 구성 사양에 맞는 사전을 반환해야 합니다.
 - `entity`:  이 스윕에서 생성되는 W&B run을 저장할 username 또는 팀 이름.  지정한 entity가 이미 존재해야 합니다. entity를 지정하지 않으면, 기본 entity(대체로 본인 username)로 저장됩니다.
 - `project`:  스윕에서 생성된 W&B run이 저장될 Project 이름.  project를 지정하지 않으면, 'Uncategorized'라는 프로젝트에 run이 저장됩니다.
 - `prior_runs`:  이 스윕에 추가할 기존 run들의 run ID 목록.



**반환값:**
 
 - `sweep_id`:  (str) 스윕에 대한 고유 식별자.
```
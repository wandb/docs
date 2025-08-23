---
title: '`Est. Runs` 열은 무엇인가요?'
menu:
  support:
    identifier: ko-support-kb-articles-est_runs_column
support:
- 스윕
- 하이퍼파라미터
toc_hide: true
type: docs
url: /support/:filename
---

W&B는 이산형 탐색 공간을 가진 W&B Sweep을 생성할 때 생성될 것으로 예상되는 Run의 개수를 제공합니다. 이 총합은 탐색 공간의 데카르트 곱을 반영합니다.

예를 들어, 다음과 같은 탐색 공간을 고려해보세요.

{{< img src="/images/sweeps/sweeps_faq_whatisestruns_1.png" alt="Estimated runs column" >}}

이 경우 데카르트 곱은 9가 됩니다. W&B App UI에서는 이 값을 예상 run 수(**Est. Runs**)로 표시합니다.

{{< img src="/images/sweeps/spaces_sweeps_faq_whatisestruns_2.webp" alt="Sweep run estimation" >}}

예상 Run 수를 프로그래밍적으로 가져오려면, W&B SDK에서 Sweep 오브젝트의 `expected_run_count` 속성을 사용하면 됩니다.

```python
sweep_id = wandb.sweep(
    sweep_configs, project="your_project_name", entity="your_entity_name"
)
api = wandb.Api()
sweep = api.sweep(f"your_entity_name/your_project_name/sweeps/{sweep_id}")
print(f"EXPECTED RUN COUNT = {sweep.expected_run_count}")
```

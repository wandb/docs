---
title: What is the `Est. Runs` column?
menu:
  support:
    identifier: ko-support-kb-articles-est_runs_column
support:
- sweeps
- hyperparameter
toc_hide: true
type: docs
url: /ko/support/:filename
translationKey: est_runs_column
---
W&B는 분리된 검색 공간으로 W&B 스윕을 생성할 때 생성되는 예상 Run 수를 제공합니다. 이 총계는 검색 공간의 데카르트 곱을 반영합니다.

예를 들어 다음 검색 공간을 고려하십시오.

{{< img src="/images/sweeps/sweeps_faq_whatisestruns_1.png" alt="" >}}

이 경우 데카르트 곱은 9와 같습니다. W&B는 이 값을 App UI에 예상 Run 수 (**Est. Runs**)로 표시합니다.

{{< img src="/images/sweeps/spaces_sweeps_faq_whatisestruns_2.webp" alt="" >}}

예상 Run 수를 프로그래밍 방식으로 검색하려면 W&B SDK 내에서 스윕 오브젝트의 `expected_run_count` 속성을 사용하십시오.

```python
sweep_id = wandb.sweep(
    sweep_configs, project="your_project_name", entity="your_entity_name"
)
api = wandb.Api()
sweep = api.sweep(f"your_entity_name/your_project_name/sweeps/{sweep_id}")
print(f"EXPECTED RUN COUNT = {sweep.expected_run_count}")
```

---
title: How can I log additional metrics after a run completes?
menu:
  support:
    identifier: ko-support-log_additional_metrics_run_completes
tags:
- runs
- metrics
toc_hide: true
type: docs
---

Experiments 를 관리하는 방법은 여러 가지가 있습니다.

복잡한 워크플로우의 경우, 여러 개의 run 을 사용하고 단일 실험 내의 모든 프로세스에 대해 [`wandb.init`]({{< relref path="/guides/models/track/launch.md" lang="ko" >}}) 의 group 파라미터를 고유한 값으로 설정합니다. [**Runs** 탭]({{< relref path="/guides/models/track/project-page.md#runs-tab" lang="ko" >}}) 은 테이블을 group ID 별로 그룹화하여 시각화가 제대로 작동하도록 합니다. 이 접근 방식을 사용하면 결과를 한 곳에 기록하면서 동시 Experiments 및 트레이닝 run 을 활성화할 수 있습니다.

더 간단한 워크플로우의 경우, `resume=True` 와 `id=UNIQUE_ID` 로 `wandb.init` 을 호출한 다음 동일한 `id=UNIQUE_ID` 로 `wandb.init` 을 다시 호출합니다. [`wandb.log`]({{< relref path="/guides/models/track/log/" lang="ko" >}}) 또는 `wandb.summary` 로 정상적으로 로그하면 run 값이 그에 따라 업데이트됩니다.

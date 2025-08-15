---
title: run 이 완료된 후 추가 메트릭을 어떻게 로그할 수 있나요?
menu:
  support:
    identifier: ko-support-kb-articles-log_additional_metrics_run_completes
support:
- run
- 메트릭
toc_hide: true
type: docs
url: /support/:filename
---

실험을 관리하는 방법에는 여러 가지가 있습니다.

복잡한 워크플로우의 경우, 여러 개의 run을 사용하고 [`wandb.init()`]({{< relref path="/guides/models/track/create-an-experiment.md" lang="ko" >}})에서 group 파라미터를 동일한 실험 내 모든 프로세스에 고유한 값으로 지정하세요. [**Runs** 탭]({{< relref path="/guides/models/track/project-page.md#runs-tab" lang="ko" >}})에서는 테이블이 group ID로 묶여, 시각화가 제대로 작동합니다. 이 방식은 여러 실험과 트레이닝 run을 동시에 실행하면서 결과를 한 위치에 로그할 수 있게 해줍니다.

간단한 워크플로우의 경우, `wandb.init()`을 `resume=True`와 `id=UNIQUE_ID`로 호출한 뒤 같은 `id=UNIQUE_ID`로 다시 `wandb.init()`을 호출하세요. [`run.log()`]({{< relref path="/guides/models/track/log/" lang="ko" >}}) 또는 `run.summary()`로 평소와 같이 로그하면, run 값이 그에 맞게 업데이트됩니다.
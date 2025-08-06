---
title: 스윕에서 run 의 모델을 가장 효과적으로 로깅하는 방법은 무엇인가요?
menu:
  support:
    identifier: ko-support-kb-articles-best_log_models_runs_sweep
support:
- Artifacts
- Sweeps
toc_hide: true
type: docs
url: /support/:filename
---

스윕에서 모델을 로그하는 효과적인 방법 중 하나는 [스윕]({{< relref path="/guides/models/sweeps/" lang="ko" >}})에 대한 모델 아티팩트를 생성하는 것입니다. 각 버전은 스윕에서 나온 서로 다른 run 을 나타냅니다. 아래와 같이 구현할 수 있습니다:

```python
# 스윕 아티팩트 생성 예시
wandb.Artifact(name="sweep_name", type="model")
```
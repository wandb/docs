---
title: How do I best log models from runs in a sweep?
menu:
  support:
    identifier: ko-support-kb-articles-best_log_models_runs_sweep
support:
- artifacts
- sweeps
toc_hide: true
type: docs
url: /support/:filename
---

[스윕]({{< relref path="/guides/models/sweeps/" lang="ko" >}})에서 모델을 로깅하는 효과적인 방법 중 하나는 스윕을 위한 모델 아티팩트를 생성하는 것입니다. 각 버전은 스윕의 서로 다른 run을 나타냅니다. 다음과 같이 구현합니다.

```python
wandb.Artifact(name="sweep_name", type="model")
```

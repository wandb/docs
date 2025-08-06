---
title: 여러 메트릭 최적화하기
menu:
  support:
    identifier: ko-support-kb-articles-optimizing_multiple_metrics
support:
- 스윕
- 메트릭
toc_hide: true
type: docs
url: /support/:filename
---

여러 개의 메트릭을 한 번의 run 에서 최적화하려면, 각 메트릭에 가중치를 곱한 값을 합산하세요.

```python
with wandb.init() as run:
  # 개별 메트릭 로그
  metric_a = run.summary.get("metric_a", 0.5)
  metric_b = run.summary.get("metric_b", 0.7)
  # 필요에 따라 다른 메트릭들도 로그
  metric_n = run.summary.get("metric_n", 0.9)

  # 메트릭에 가중치 부여해서 결합하기
  # 최적화 목표에 따라 가중치를 조절하세요
  # 예를 들어, metric_a 와 metric_n 에 더 큰 비중을 두고 싶다면:  
  metric_combined = 0.3 * metric_a + 0.2 * metric_b + ... + 1.5 * metric_n
  run.log({"metric_combined": metric_combined})
```

새로 결합된 메트릭을 로그하고, 이를 최적화 목표로 설정하세요:

```yaml
metric:
  name: metric_combined
  goal: minimize
```
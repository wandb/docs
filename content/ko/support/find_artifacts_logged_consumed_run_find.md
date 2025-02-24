---
title: How can I find the artifacts logged or consumed by a run? How can I find the
  runs that produced or consumed an artifact?
menu:
  support:
    identifier: ko-support-find_artifacts_logged_consumed_run_find
tags:
- artifacts
toc_hide: true
type: docs
---

W&B는 각 run에 의해 로깅된 아티팩트와 각 run에서 사용하여 아티팩트 그래프를 구성한 아티팩트를 추적합니다. 이 그래프는 run과 아티팩트를 나타내는 노드가 있는 이분 방향성 비순환 그래프입니다. 예제는 [여기](https://wandb.ai/shawn/detectron2-11/artifacts/dataset/furniture-small-val/06d5ddd4deeb2a6ebdd5/graph)에서 볼 수 있습니다. ("Explode"를 클릭하여 그래프를 확장하십시오).

Public API를 사용하여 아티팩트 또는 run에서 시작하여 프로그래밍 방식으로 그래프를 탐색합니다.

{{< tabpane text=true >}}
{{% tab "아티팩트에서" %}}

```python
api = wandb.Api()

artifact = api.artifact("project/artifact:alias")

# 아티팩트에서 그래프 위로 이동:
producer_run = artifact.logged_by()
# 아티팩트에서 그래프 아래로 이동:
consumer_runs = artifact.used_by()

# Run에서 그래프 아래로 이동:
next_artifacts = consumer_runs[0].logged_artifacts()
# Run에서 그래프 위로 이동:
previous_artifacts = producer_run.used_artifacts()
```

{{% /tab %}}
{{% tab "Run에서" %}}

```python
api = wandb.Api()

run = api.run("entity/project/run_id")

# Run에서 그래프 아래로 이동:
produced_artifacts = run.logged_artifacts()
# Run에서 그래프 위로 이동:
consumed_artifacts = run.used_artifacts()

# 아티팩트에서 그래프 위로 이동:
earlier_run = consumed_artifacts[0].logged_by()
# 아티팩트에서 그래프 아래로 이동:
consumer_runs = produced_artifacts[0].used_by()
```

{{% /tab %}}
{{% /tabpane %}}

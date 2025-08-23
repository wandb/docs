---
title: run 이 로그하거나 사용한 Artifacts 를 어떻게 찾을 수 있나요? 또는 특정 Artifacts 를 생성하거나 사용한 run 들을
  어떻게 찾을 수 있나요?
menu:
  support:
    identifier: ko-support-kb-articles-find_artifacts_logged_consumed_run_find
support:
- 아티팩트
toc_hide: true
type: docs
url: /support/:filename
---

W&B는 각 run이 로그한 Artifacts와 각 run에서 사용된 Artifacts를 추적하여 아티팩트 그래프를 만듭니다. 이 그래프는 run과 Artifacts를 노드로 가지는 이분, 방향성, 비순환 그래프입니다. 예시는 [여기](https://wandb.ai/shawn/detectron2-11/artifacts/dataset/furniture-small-val/06d5ddd4deeb2a6ebdd5/graph)에서 확인할 수 있습니다(그래프를 확장하려면 "Explode"를 클릭하세요).

Public API를 사용하면 artifact나 run 중 원하는 지점에서 그래프를 프로그래밍적으로 탐색할 수 있습니다.

{{< tabpane text=true >}}
{{% tab "Artifact에서 시작" %}}

```python
api = wandb.Api()

artifact = api.artifact("project/artifact:alias")

# artifact에서 그래프를 위로 타고 올라감:
producer_run = artifact.logged_by()
# artifact에서 그래프를 아래로 내려감:
consumer_runs = artifact.used_by()

# run에서 그래프를 아래로 내려감:
next_artifacts = consumer_runs[0].logged_artifacts()
# run에서 그래프를 위로 타고 올라감:
previous_artifacts = producer_run.used_artifacts()
```

{{% /tab %}}
{{% tab "Run에서 시작" %}}

```python
api = wandb.Api()

run = api.run("entity/project/run_id")

# run에서 그래프를 아래로 내려감:
produced_artifacts = run.logged_artifacts()
# run에서 그래프를 위로 타고 올라감:
consumed_artifacts = run.used_artifacts()

# artifact에서 그래프를 위로 타고 올라감:
earlier_run = consumed_artifacts[0].logged_by()
# artifact에서 그래프를 아래로 내려감:
consumer_runs = produced_artifacts[0].used_by()
```

{{% /tab %}}
{{% /tabpane %}}
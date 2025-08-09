---
title: Summary 메트릭 로그
menu:
  default:
    identifier: ko-guides-models-track-log-log-summary
    parent: log-objects-and-media
---

트레이닝 중에 시간이 지남에 따라 변하는 값 외에도, 모델이나 전처리 단계 전체를 요약하는 단일 값을 추적하는 것이 중요한 경우가 많습니다. 이러한 정보는 W&B Run의 `summary` 딕셔너리에 로그로 남길 수 있습니다. Run의 summary 딕셔너리는 numpy 배열, PyTorch 텐서, TensorFlow 텐서를 모두 처리할 수 있습니다. 값이 이들 타입 중 하나라면 전체 텐서를 바이너리 파일로 저장하고, min, mean, variance, percentiles 등 주요 메트릭은 summary 오브젝트에 저장됩니다.

`wandb.Run.log()`으로 마지막으로 기록된 값은 W&B Run의 summary 딕셔너리에 자동으로 설정됩니다. summary 메트릭 딕셔너리를 수정하면 이전 값이 덮어씌워집니다.

아래 코드 예시는 W&B에 커스텀 summary 메트릭을 제공하는 방법을 보여줍니다:

```python
import wandb
import argparse

with wandb.init(config=args) as run:
  best_accuracy = 0
  for epoch in range(1, args.epochs + 1):
      test_loss, test_accuracy = test()
      if test_accuracy > best_accuracy:
          run.summary["best_accuracy"] = test_accuracy
          best_accuracy = test_accuracy
```

트레이닝이 끝난 뒤에도 기존 W&B Run의 summary 속성을 업데이트할 수 있습니다. [W&B Public API]({{< relref path="/ref/python/public-api/" lang="ko" >}})를 사용해 summary 속성을 업데이트하세요:

```python
api = wandb.Api()
run = api.run("username/project/run_id")
run.summary["tensor"] = np.random.random(1000)
run.summary.update()
```

## summary 메트릭 커스터마이즈하기

커스텀 summary 메트릭은 `run.summary`에서 트레이닝 중 가장 좋은 단계의 모델 성능을 추적하는 데 유용합니다. 예를 들어, 마지막 값 대신 최대 정확도나 최소 손실 값을 기록하고 싶을 수 있습니다.

기본적으로 summary는 이력(history)에서 마지막 값을 사용합니다. summary 메트릭을 커스터마이즈하려면, `define_metric`에서 `summary` 인수를 전달하세요. 다음 값 중 하나를 사용할 수 있습니다:

* `"min"`
* `"max"`
* `"mean"`
* `"best"`
* `"last"`
* `"none"`

`"best"`는 옵션으로 `objective` 인수를 `"minimize"` 또는 `"maximize"`로 설정해야 사용할 수 있습니다.

아래 예시는 loss와 accuracy에 대해 최소/최대 값을 summary에 추가하는 방법입니다:

```python
import wandb
import random

random.seed(1)

with wandb.init() as run:
    # 손실의 최소/최대 summary 값 정의
    run.define_metric("loss", summary="min")
    run.define_metric("loss", summary="max")

    # 정확도의 최소/최대 summary 값 정의
    run.define_metric("acc", summary="min")
    run.define_metric("acc", summary="max")

    for i in range(10):
        log_dict = {
            "loss": random.uniform(0, 1 / (i + 1)),
            "acc": random.uniform(1 / (i + 1), 1),
        }
        run.log(log_dict)
```

## summary 메트릭 확인하기

summary 값은 run의 **Overview** 페이지나 프로젝트의 runs 테이블에서 확인할 수 있습니다.

{{< tabpane text=true >}}
{{% tab header="Run Overview" value="overview" %}}

1. W&B App에 접속합니다.
2. **Workspace** 탭을 선택합니다.
3. runs 목록에서 summary 값을 로그한 run 이름을 클릭합니다.
4. **Overview** 탭을 선택합니다.
5. **Summary** 섹션에서 summary 값을 확인하세요.

{{< img src="/images/track/customize_summary.png" alt="Run overview" >}}

{{% /tab %}}
{{% tab header="Run Table" value="run table" %}}

1. W&B App에 접속합니다.
2. **Runs** 탭을 선택합니다.
3. runs 테이블에서 summary 값들이 해당 이름별 컬럼에 표시됩니다.

{{% /tab %}}

{{% tab header="W&B Public API" value="api" %}}

W&B Public API를 사용해서 run의 summary 값을 가져올 수 있습니다.

아래 예시는 W&B Public API와 pandas로 특정 run에 기록된 summary 값을 가져오는 한 방법입니다:

```python
import wandb
import pandas

entity = "<your-entity>"
project = "<your-project>"
run_name = "<your-run-name>" # summary 값을 가진 run의 이름

all_runs = []

for run in api.runs(f"{entity}/{project_name}"):
    print("Fetching details for run: ", run.id, run.name)
    run_data = {
              "id": run.id,
              "name": run.name,
              "url": run.url,
              "state": run.state,
              "tags": run.tags,
              "config": run.config,
              "created_at": run.created_at,
              "system_metrics": run.system_metrics,
              "summary": run.summary,
              "project": run.project,
              "entity": run.entity,
              "user": run.user,
              "path": run.path,
              "notes": run.notes,
              "read_only": run.read_only,
              "history_keys": run.history_keys,
              "metadata": run.metadata,
          }
    all_runs.append(run_data)
  
# DataFrame으로 변환  
df = pd.DataFrame(all_runs)

# run 이름으로 행을 골라 summary를 딕셔너리로 반환
df[df['name']==run_name].summary.reset_index(drop=True).to_dict()
```

{{% /tab %}}
{{< /tabpane >}}
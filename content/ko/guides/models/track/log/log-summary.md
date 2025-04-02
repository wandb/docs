---
title: Log summary metrics
menu:
  default:
    identifier: ko-guides-models-track-log-log-summary
    parent: log-objects-and-media
---

트레이닝 과정에서 시간이 지남에 따라 변하는 값 외에도, 모델 또는 전처리 단계를 요약하는 단일 값을 추적하는 것이 중요한 경우가 많습니다. W&B Run의 `summary` 사전에 이 정보를 기록하세요. Run의 summary 사전은 numpy 배열, PyTorch 텐서 또는 TensorFlow 텐서를 처리할 수 있습니다. 값이 이러한 유형 중 하나인 경우 전체 텐서를 바이너리 파일에 유지하고 요약 오브젝트에 최소값, 평균, 분산, 백분위수 등과 같은 높은 수준의 메트릭을 저장합니다.

`wandb.log`로 기록된 마지막 값은 W&B Run에서 자동으로 summary 사전으로 설정됩니다. summary 메트릭 사전이 수정되면 이전 값은 손실됩니다.

다음 코드 조각은 사용자 정의 summary 메트릭을 W&B에 제공하는 방법을 보여줍니다.
```python
wandb.init(config=args)

best_accuracy = 0
for epoch in range(1, args.epochs + 1):
    test_loss, test_accuracy = test()
    if test_accuracy > best_accuracy:
        wandb.summary["best_accuracy"] = test_accuracy
        best_accuracy = test_accuracy
```

트레이닝이 완료된 후 기존 W&B Run의 summary 속성을 업데이트할 수 있습니다. [W&B Public API]({{< relref path="/ref/python/public-api/" lang="ko" >}})를 사용하여 summary 속성을 업데이트합니다.

```python
api = wandb.Api()
run = api.run("username/project/run_id")
run.summary["tensor"] = np.random.random(1000)
run.summary.update()
```

## summary 메트릭 사용자 정의

사용자 정의 summary 메트릭은 `wandb.summary`에서 트레이닝의 최적 단계에서 모델 성능을 캡처하는 데 유용합니다. 예를 들어 최종 값 대신 최대 정확도 또는 최소 손실 값을 캡처할 수 있습니다.

기본적으로 summary는 히스토리의 최종 값을 사용합니다. summary 메트릭을 사용자 정의하려면 `define_metric`에서 `summary` 인수를 전달합니다. 다음 값을 사용할 수 있습니다.

* `"min"`
* `"max"`
* `"mean"`
* `"best"`
* `"last"`
* `"none"`

선택적 `objective` 인수를 `"minimize"` 또는 `"maximize"`로 설정한 경우에만 `"best"`를 사용할 수 있습니다.

다음 예제는 손실 및 정확도의 최소값과 최대값을 summary에 추가합니다.

```python
import wandb
import random

random.seed(1)
wandb.init()

# 손실에 대한 최소값 및 최대값 summary
wandb.define_metric("loss", summary="min")
wandb.define_metric("loss", summary="max")

# 정확도에 대한 최소값 및 최대값 summary
wandb.define_metric("acc", summary="min")
wandb.define_metric("acc", summary="max")

for i in range(10):
    log_dict = {
        "loss": random.uniform(0, 1 / (i + 1)),
        "acc": random.uniform(1 / (i + 1), 1),
    }
    wandb.log(log_dict)
```

## summary 메트릭 보기

run의 **Overview** 페이지 또는 프로젝트의 runs 테이블에서 summary 값을 봅니다.

{{< tabpane text=true >}}
{{% tab header="Run Overview" value="overview" %}}

1. W&B 앱으로 이동합니다.
2. **Workspace** 탭을 선택합니다.
3. runs 목록에서 summary 값이 기록된 run의 이름을 클릭합니다.
4. **Overview** 탭을 선택합니다.
5. **Summary** 섹션에서 summary 값을 봅니다.

{{< img src="/images/track/customize_summary.png" alt="W&B에 기록된 run의 Overview 페이지. UI의 오른쪽 하단 모서리에 Summary 메트릭 섹션 내에서 기계 학습 모델 정확도 및 손실의 최소값과 최대값이 표시됩니다." >}}

{{% /tab %}}
{{% tab header="Run Table" value="run table" %}}

1. W&B 앱으로 이동합니다.
2. **Runs** 탭을 선택합니다.
3. runs 테이블 내에서 summary 값의 이름을 기준으로 열 내에서 summary 값을 볼 수 있습니다.

{{% /tab %}}

{{% tab header="W&B Public API" value="api" %}}

W&B Public API를 사용하여 run의 summary 값을 가져올 수 있습니다.

다음 코드 예제는 W&B Public API 및 pandas를 사용하여 특정 run에 기록된 summary 값을 검색하는 한 가지 방법을 보여줍니다.

```python
import wandb
import pandas

entity = "<your-entity>"
project = "<your-project>"
run_name = "<your-run-name>" # summary 값이 있는 run의 이름

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

# 열 이름(run)을 기준으로 행을 가져오고 사전으로 변환
df[df['name']==run_name].summary.reset_index(drop=True).to_dict()
```

{{% /tab %}}
{{< /tabpane >}}

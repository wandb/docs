---
title: Import and export data
description: MLFlow에서 데이터를 가져오고, W\&B에 저장한 데이터를 내보내거나 업데이트합니다.
menu:
  default:
    identifier: ko-guides-models-track-public-api-guide
    parent: experiments
weight: 8
---

W&B Public API를 사용하여 데이터를 내보내거나 가져올 수 있습니다.

{{% alert %}}
이 기능은 python>=3.8이 필요합니다.
{{% /alert %}}

## MLFlow에서 데이터 가져오기

W&B는 Experiments, Runs, Artifacts, 메트릭 및 기타 메타데이터를 포함하여 MLFlow에서 데이터를 가져오는 것을 지원합니다.

다음과 같이 종속성을 설치합니다:

```shell
# 참고: py38+ 필요
pip install wandb[importers]
```

W&B에 로그인합니다. 이전에 로그인하지 않은 경우 프롬프트에 따르십시오.

```shell
wandb login
```

기존 MLFlow 서버에서 모든 Runs을 가져옵니다:

```py
from wandb.apis.importers.mlflow import MlflowImporter

importer = MlflowImporter(mlflow_tracking_uri="...")

runs = importer.collect_runs()
importer.import_runs(runs)
```

기본적으로 `importer.collect_runs()`는 MLFlow 서버에서 모든 Runs을 수집합니다. 특정 서브셋을 업로드하려면 Runs의 반복 가능한 객체를 직접 구성하여 임포터에 전달할 수 있습니다.

```py
import mlflow
from wandb.apis.importers.mlflow import MlflowRun

client = mlflow.tracking.MlflowClient(mlflow_tracking_uri)

runs: Iterable[MlflowRun] = []
for run in mlflow_client.search_runs(...):
    runs.append(MlflowRun(run, client))

importer.import_runs(runs)
```

{{% alert %}}
Databricks MLFlow에서 가져오는 경우 먼저 [Databricks CLI를 구성](https://docs.databricks.com/dev-tools/cli/index.html)해야 할 수 있습니다.

이전 단계에서 `mlflow-tracking-uri="databricks"`를 설정하세요.
{{% /alert %}}

Artifacts 가져오기를 생략하려면 `artifacts=False`를 전달하면 됩니다:

```py
importer.import_runs(runs, artifacts=False)
```

특정 W&B 엔티티 및 프로젝트로 가져오려면 `Namespace`를 전달하면 됩니다:

```py
from wandb.apis.importers import Namespace

importer.import_runs(runs, namespace=Namespace(entity, project))
```

## 데이터 내보내기

Public API를 사용하여 W&B에 저장한 데이터를 내보내거나 업데이트합니다. 이 API를 사용하기 전에 스크립트에서 데이터를 기록하십시오. 자세한 내용은 [퀵스타트]({{< relref path="/guides/quickstart.md" lang="ko" >}})를 확인하세요.

**Public API 유스 케이스**

- **데이터 내보내기**: Jupyter Notebook에서 사용자 정의 분석을 위해 데이터프레임을 가져옵니다. 데이터를 탐색한 후 새로운 분석 Run을 생성하고 결과를 기록하여 결과를 동기화할 수 있습니다(예: `wandb.init(job_type="analysis")`).
- **기존 Runs 업데이트**: W&B Run과 연결하여 기록된 데이터를 업데이트할 수 있습니다. 예를 들어 아키텍처 또는 원래 기록되지 않은 하이퍼파라미터와 같은 추가 정보를 포함하도록 Runs 집합의 구성을 업데이트할 수 있습니다.

사용 가능한 함수에 대한 자세한 내용은 [생성된 참조 문서]({{< relref path="/ref/python/public-api/" lang="ko" >}})를 참조하십시오.

### API 키 생성

API 키는 W&B에 대한 컴퓨터 인증을 처리합니다. 사용자 프로필에서 API 키를 생성할 수 있습니다.

{{% alert %}}
보다 간소화된 접근 방식을 위해 [https://wandb.ai/authorize](https://wandb.ai/authorize)로 직접 이동하여 API 키를 생성할 수 있습니다. 표시된 API 키를 복사하여 비밀번호 관리자와 같은 안전한 위치에 저장합니다.
{{% /alert %}}

1. 오른쪽 상단 모서리에 있는 사용자 프로필 아이콘을 클릭합니다.
2. **사용자 설정**을 선택한 다음 **API 키** 섹션으로 스크롤합니다.
3. **표시**를 클릭합니다. 표시된 API 키를 복사합니다. API 키를 숨기려면 페이지를 새로 고칩니다.

### Run 경로 찾기

Public API를 사용하려면 `<entity>/<project>/<run_id>`인 Run 경로가 필요한 경우가 많습니다. 앱 UI에서 Run 페이지를 열고 [Overview 탭]({{< relref path="/guides/models/track/runs/#overview-tab" lang="ko" >}})을 클릭하여 Run 경로를 가져옵니다.

### Run 데이터 내보내기

완료되었거나 활성 상태인 Run에서 데이터를 다운로드합니다. 일반적인 사용 사례로는 Jupyter 노트북에서 사용자 정의 분석을 위해 데이터프레임을 다운로드하거나 자동화된 환경에서 사용자 정의 로직을 사용하는 것이 있습니다.

```python
import wandb

api = wandb.Api()
run = api.run("<entity>/<project>/<run_id>")
```

Run 객체의 가장 일반적으로 사용되는 속성은 다음과 같습니다:

| 속성          | 의미                                                                                                                                                                                                                                                                                                                          |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `run.config`    | 트레이닝 Run의 하이퍼파라미터 또는 데이터셋 Artifact를 만드는 Run의 전처리 방법과 같은 Run의 구성 정보 사전입니다. 이를 Run의 입력이라고 생각하십시오.                                                                                                                                                               |
| `run.history()` | 손실과 같이 모델이 트레이닝되는 동안 변하는 값을 저장하기 위한 사전 목록입니다. `wandb.log()` 코맨드는 이 객체에 추가됩니다.                                                                                                                                                                                           |
| `run.summary`   | Run 결과의 요약 정보 사전입니다. 여기에는 정확도 및 손실과 같은 스칼라 또는 큰 파일이 포함될 수 있습니다. 기본적으로 `wandb.log()`는 요약을 기록된 시계열의 최종 값으로 설정합니다. 요약 내용은 직접 설정할 수도 있습니다. 요약을 Run의 출력이라고 생각하십시오.                                                                                                 |

과거 Runs의 데이터를 수정하거나 업데이트할 수도 있습니다. 기본적으로 API 객체의 단일 인스턴스는 모든 네트워크 요청을 캐시합니다. 유스 케이스에서 실행 중인 스크립트의 실시간 정보가 필요한 경우 `api.flush()`를 호출하여 업데이트된 값을 가져옵니다.

### 다양한 속성 이해

아래의 Run의 경우

```python
n_epochs = 5
config = {"n_epochs": n_epochs}
run = wandb.init(project=project, config=config)
for n in range(run.config.get("n_epochs")):
    run.log(
        {"val": random.randint(0, 1000), "loss": (random.randint(0, 1000) / 1000.00)}
    )
run.finish()
```

다음은 위의 Run 객체 속성에 대한 다양한 출력입니다

#### `run.config`

```python
{"n_epochs": 5}
```

#### `run.history()`

```shell
   _step  val   loss  _runtime  _timestamp
0      0  500  0.244         4  1644345412
1      1   45  0.521         4  1644345412
2      2  240  0.785         4  1644345412
3      3   31  0.305         4  1644345412
4      4  525  0.041         4  1644345412
```

#### `run.summary`

```python
{
    "_runtime": 4,
    "_step": 4,
    "_timestamp": 1644345412,
    "_wandb": {"runtime": 3},
    "loss": 0.041,
    "val": 525,
}
```

### 샘플링

기본 히스토리 메서드는 메트릭을 고정된 수의 샘플로 샘플링합니다(기본값은 500이며, `samples` __ 인수로 변경할 수 있음). 대규모 Run에서 모든 데이터를 내보내려면 `run.scan_history()` 메서드를 사용하면 됩니다. 자세한 내용은 [API 참조]({{< relref path="/ref/python/public-api" lang="ko" >}})를 참조하십시오.

### 여러 Runs 쿼리

{{< tabpane text=true >}}
    {{% tab header="DataFrame 및 CSV" %}}
이 예제 스크립트는 프로젝트를 찾고 이름, 구성 및 요약 통계가 있는 Runs의 CSV를 출력합니다. `<entity>` 및 `<project>`를 W&B 엔티티 및 프로젝트 이름으로 각각 바꿉니다.

```python
import pandas as pd
import wandb

api = wandb.Api()
entity, project = "<entity>", "<project>"
runs = api.runs(entity + "/" + project)

summary_list, config_list, name_list = [], [], []
for run in runs:
    # .summary에는 정확도와 같은
    # 메트릭에 대한 출력 키/값이 포함되어 있습니다.
    #  큰 파일을 생략하기 위해 ._json_dict를 호출합니다
    summary_list.append(run.summary._json_dict)

    # .config에는 하이퍼파라미터가 포함되어 있습니다.
    #  _로 시작하는 특수 값을 제거합니다.
    config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

    # .name은 Run의 사람이 읽을 수 있는 이름입니다.
    name_list.append(run.name)

runs_df = pd.DataFrame(
    {"summary": summary_list, "config": config_list, "name": name_list}
)

runs_df.to_csv("project.csv")
```
    {{% /tab %}}
    {{% tab header="MongoDB 스타일" %}}
W&B API는 api.runs()를 사용하여 프로젝트의 Runs을 쿼리하는 방법도 제공합니다. 가장 일반적인 유스 케이스는 사용자 정의 분석을 위해 Runs 데이터를 내보내는 것입니다. 쿼리 인터페이스는 [MongoDB에서 사용하는 인터페이스](https://docs.mongodb.com/manual/reference/operator/query)와 동일합니다.

```python
runs = api.runs(
    "username/project",
    {"$or": [{"config.experiment_name": "foo"}, {"config.experiment_name": "bar"}]},
)
print(f"Found {len(runs)} runs")
```    
    {{% /tab %}}
{{< /tabpane >}}

`api.runs`를 호출하면 반복 가능하고 목록처럼 작동하는 `Runs` 객체가 반환됩니다. 기본적으로 객체는 필요에 따라 한 번에 50개의 Runs을 순서대로 로드하지만 `per_page` 키워드 인수를 사용하여 페이지당 로드되는 수를 변경할 수 있습니다.

`api.runs`는 `order` 키워드 인수도 허용합니다. 기본 순서는 `-created_at`입니다. 결과를 오름차순으로 정렬하려면 `+created_at`를 지정합니다. 구성 또는 요약 값으로 정렬할 수도 있습니다. 예를 들어 `summary.val_acc` 또는 `config.experiment_name`입니다.

### 오류 처리

W&B 서버와 통신하는 동안 오류가 발생하면 `wandb.CommError`가 발생합니다. 원래 예외는 `exc` 속성을 통해 조사할 수 있습니다.

### API를 통해 최신 git 커밋 가져오기

UI에서 Run을 클릭한 다음 Run 페이지에서 Overview 탭을 클릭하여 최신 git 커밋을 확인합니다. 또한 `wandb-metadata.json` 파일에도 있습니다. Public API를 사용하면 `run.commit`으로 git 해시를 가져올 수 있습니다.

### Run 중 Run의 이름 및 ID 가져오기

`wandb.init()`를 호출한 후 다음과 같이 스크립트에서 임의 Run ID 또는 사람이 읽을 수 있는 Run 이름을 액세스할 수 있습니다.

- 고유 Run ID(8자 해시): `wandb.run.id`
- 임의 Run 이름(사람이 읽을 수 있음): `wandb.run.name`

Runs에 유용한 식별자를 설정하는 방법을 고려하고 있다면 다음을 권장합니다.

- **Run ID**: 생성된 해시로 둡니다. 이는 프로젝트의 Runs에서 고유해야 합니다.
- **Run 이름**: 차트에서 여러 줄의 차이점을 알 수 있도록 짧고 읽기 쉽고 가급적이면 고유해야 합니다.
- **Run 노트**: Run에서 수행하는 작업에 대한 간단한 설명을 적어두는 것이 좋습니다. `wandb.init(notes="여기에 메모 입력")`로 설정할 수 있습니다.
- **Run 태그**: Run 태그에서 동적으로 추적하고 UI에서 필터를 사용하여 테이블을 원하는 Runs로 필터링합니다. 스크립트에서 태그를 설정한 다음 Runs 테이블과 Run 페이지의 Overview 탭 모두에서 UI에서 편집할 수 있습니다. 자세한 내용은 [여기]({{< relref path="/guides/models/track/runs/tags.md" lang="ko" >}})의 자세한 지침을 참조하십시오.

## Public API 예제

### matplotlib 또는 seaborn에서 시각화하기 위해 데이터 내보내기

몇 가지 일반적인 내보내기 패턴은 [API 예제]({{< relref path="/ref/python/public-api/" lang="ko" >}})를 확인하십시오. 사용자 정의 플롯 또는 확장된 Runs 테이블에서 다운로드 버튼을 클릭하여 브라우저에서 CSV를 다운로드할 수도 있습니다.

### Run에서 메트릭 읽기

이 예제는 `wandb.log({"accuracy": acc})`로 저장된 Run에 대해 `"<entity>/<project>/<run_id>"`에 저장된 타임스탬프 및 정확도를 출력합니다.

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
if run.state == "finished":
    for i, row in run.history().iterrows():
        print(row["_timestamp"], row["accuracy"])
```

### Runs 필터링

MongoDB Query Language를 사용하여 필터링할 수 있습니다.

#### 날짜

```python
runs = api.runs(
    "<entity>/<project>",
    {"$and": [{"created_at": {"$lt": "YYYY-MM-DDT##", "$gt": "YYYY-MM-DDT##"}}]},
)
```

### Run에서 특정 메트릭 읽기

Run에서 특정 메트릭을 가져오려면 `keys` 인수를 사용합니다. `run.history()`를 사용할 때 기본 샘플 수는 500입니다. 특정 메트릭을 포함하지 않는 기록된 단계는 출력 데이터프레임에 `NaN`으로 표시됩니다. `keys` 인수를 사용하면 API가 나열된 메트릭 키를 포함하는 단계를 더 자주 샘플링합니다.

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
if run.state == "finished":
    for i, row in run.history(keys=["accuracy"]).iterrows():
        print(row["_timestamp"], row["accuracy"])
```

### 두 Runs 비교

이렇게 하면 `run1`과 `run2` 간에 다른 구성 파라미터가 출력됩니다.

```python
import pandas as pd
import wandb

api = wandb.Api()

# <entity>, <project> 및 <run_id>로 바꿉니다
run1 = api.run("<entity>/<project>/<run_id>")
run2 = api.run("<entity>/<project>/<run_id>")


df = pd.DataFrame([run1.config, run2.config]).transpose()

df.columns = [run1.name, run2.name]
print(df[df[run1.name] != df[run2.name]])
```

출력:

```
              c_10_sgd_0.025_0.01_long_switch base_adam_4_conv_2fc
batch_size                                 32                   16
n_conv_layers                               5                    4
optimizer                             rmsprop                 adam
```

### Run이 완료된 후 Run에 대한 메트릭 업데이트

이 예제는 이전 Run의 정확도를 `0.9`로 설정합니다. 또한 이전 Run의 정확도 히스토그램을 `numpy_array`의 히스토그램으로 수정합니다.

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.summary["accuracy"] = 0.9
run.summary["accuracy_histogram"] = wandb.Histogram(numpy_array)
run.summary.update()
```

### 완료된 Run에서 메트릭 이름 바꾸기

이 예제는 테이블에서 요약 열의 이름을 바꿉니다.

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.summary["new_name"] = run.summary["old_name"]
del run.summary["old_name"]
run.summary.update()
```

{{% alert %}}
열 이름 바꾸기는 테이블에만 적용됩니다. 차트는 여전히 원래 이름으로 메트릭을 참조합니다.
{{% /alert %}}

### 기존 Run에 대한 구성 업데이트

이 예제는 구성 설정 중 하나를 업데이트합니다.

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.config["key"] = updated_value
run.update()
```

### 시스템 리소스 소비를 CSV 파일로 내보내기

아래 코드 조각은 시스템 리소스 소비를 찾은 다음 CSV에 저장합니다.

```python
import wandb

run = wandb.Api().run("<entity>/<project>/<run_id>")

system_metrics = run.history(stream="events")
system_metrics.to_csv("sys_metrics.csv")
```

### 샘플링되지 않은 메트릭 데이터 가져오기

히스토리에서 데이터를 가져올 때 기본적으로 500포인트로 샘플링됩니다. `run.scan_history()`를 사용하여 기록된 모든 데이터 포인트를 가져옵니다. 다음은 히스토리에 기록된 모든 `loss` 데이터 포인트를 다운로드하는 예입니다.

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
history = run.scan_history()
losses = [row["loss"] for row in history]
```

### 히스토리에서 페이지가 매겨진 데이터 가져오기

백엔드에서 메트릭을 느리게 가져오거나 API 요청 시간이 초과되는 경우 `scan_history`에서 페이지 크기를 줄여 개별 요청 시간이 초과되지 않도록 할 수 있습니다. 기본 페이지 크기는 500이므로 다양한 크기를 실험하여 가장 적합한 크기를 확인할 수 있습니다.

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.scan_history(keys=sorted(cols), page_size=100)
```

### 프로젝트의 모든 Runs에서 메트릭을 CSV 파일로 내보내기

이 스크립트는 프로젝트에서 Runs을 가져오고 이름, 구성 및 요약 통계를 포함한 Runs의 데이터프레임과 CSV를 생성합니다. `<entity>` 및 `<project>`를 W&B 엔티티 및 프로젝트 이름으로 각각 바꿉니다.

```python
import pandas as pd
import wandb

api = wandb.Api()
entity, project = "<entity>", "<project>"
runs = api.runs(entity + "/" + project)

summary_list, config_list, name_list = [], [], []
for run in runs:
    # .summary에는 출력 키/값이 포함되어 있습니다.
    # 정확도와 같은 메트릭의 경우.
    #  큰 파일을 생략하기 위해 ._json_dict를 호출합니다
    summary_list.append(run.summary._json_dict)

    # .config에는 하이퍼파라미터가 포함되어 있습니다.
    #  _로 시작하는 특수 값을 제거합니다.
    config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

    # .name은 Run의 사람이 읽을 수 있는 이름입니다.
    name_list.append(run.name)

runs_df = pd.DataFrame(
    {"summary": summary_list, "config": config_list, "name": name_list}
)

runs_df.to_csv("project.csv")
```

### Run의 시작 시간 가져오기

이 코드 조각은 Run이 생성된 시간을 검색합니다.

```python
import wandb

api = wandb.Api()

run = api.run("entity/project/run_id")
start_time = run.created_at
```

### 완료된 Run에 파일 업로드

아래 코드 조각은 선택한 파일을 완료된 Run에 업로드합니다.

```python
import wandb

api = wandb.Api()

run = api.run("entity/project/run_id")
run.upload_file("file_name.extension")
```

### Run에서 파일 다운로드

이것은 cifar 프로젝트에서 Run ID uxte44z7과 연결된 파일 "model-best.h5"를 찾아 로컬에 저장합니다.

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.file("model-best.h5").download()
```

### Run에서 모든 파일 다운로드

이것은 Run과 연결된 모든 파일을 찾아 로컬에 저장합니다.

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
for file in run.files():
    file.download()
```

### 특정 스위프에서 Runs 가져오기

이 코드 조각은 특정 스위프와 연결된 모든 Runs을 다운로드합니다.

```python
import wandb

api = wandb.Api()

sweep = api.sweep("<entity>/<project>/<sweep_id>")
sweep_runs = sweep.runs
```

### 스위프에서 가장 적합한 Run 가져오기

다음 코드 조각은 지정된 스위프에서 가장 적합한 Run을 가져옵니다.

```python
import wandb

api = wandb.Api()

sweep = api.sweep("<entity>/<project>/<sweep_id>")
best_run = sweep.best_run()
```

`best_run`은 스위프 구성의 `metric` 파라미터에 의해 정의된 가장 적합한 메트릭을 가진 Run입니다.

### 스위프에서 가장 적합한 모델 파일 다운로드

이 코드 조각은 모델 파일을 `model.h5`에 저장한 Runs이 있는 스위프에서 가장 높은 검증 정확도를 가진 모델 파일을 다운로드합니다.

```python
import wandb

api = wandb.Api()

sweep = api.sweep("<entity>/<project>/<sweep_id>")
runs = sorted(sweep.runs, key=lambda run: run.summary.get("val_acc", 0), reverse=True)
val_acc = runs[0].summary.get("val_acc", 0)
print(f"가장 적합한 Run {runs[0].name} (검증 정확도 {val_acc}%)")

runs[0].file("model.h5").download(replace=True)
print("가장 적합한 모델이 model-best.h5에 저장되었습니다.")
```

### Run에서 지정된 확장명을 가진 모든 파일 삭제

이 코드 조각은 Run에서 지정된 확장명을 가진 파일을 삭제합니다.

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")

extension = ".png"
files = run.files()
for file in files:
    if file.name.endswith(extension):
        file.delete()
```

### 시스템 메트릭 데이터 다운로드

이 코드 조각은 Run에 대한 모든 시스템 리소스 소비 메트릭이 포함된 데이터프레임을 생성한 다음 CSV에 저장합니다.

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
system_metrics = run.history(stream="events")
system_metrics.to_csv("sys_metrics.csv")
```

### 요약 메트릭 업데이트

사전을 전달하여 요약 메트릭을 업데이트할 수 있습니다.

```python
summary.update({"key": val})
```

### Run을 실행한 코맨드 가져오기

각 Run은 Run 개요 페이지에서 실행을 시작한 코맨드를 캡처합니다. API에서 이 코맨드를 가져오려면 다음을 실행할 수 있습니다.

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")

meta = json.load(run.file("wandb-metadata.json").download())
program = ["python"] + [meta["program"]] + meta["args"]
```

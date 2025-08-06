---
title: 데이터 가져오기 및 내보내기
description: MLFlow에서 데이터 를 가져오거나, W&B에 저장한 데이터 를 내보내거나 업데이트하세요.
menu:
  default:
    identifier: ko-guides-models-track-public-api-guide
    parent: experiments
weight: 8
---

W&B Public API를 사용하여 데이터를 내보내거나 가져올 수 있습니다.

{{% alert %}}
이 기능은 python>=3.8 이 필요합니다.
{{% /alert %}}

## MLFlow에서 데이터 가져오기

W&B는 Experiments, Runs, Artifacts, 메트릭, 기타 메타데이터 등 MLFlow의 데이터를 가져오는 기능을 지원합니다.

의존성 설치:

```shell
# 주의: python 3.8 이상이 필요합니다.
pip install wandb[importers]
```

W&B에 로그인합니다. 이전에 로그인한 적 없다면 안내에 따라 로그인하세요.

```shell
wandb login
```

기존 MLFlow 서버에서 모든 run을 가져옵니다:

```py
from wandb.apis.importers.mlflow import MlflowImporter

importer = MlflowImporter(mlflow_tracking_uri="...")

runs = importer.collect_runs()
importer.import_runs(runs)
```

기본적으로 `importer.collect_runs()`는 MLFlow 서버의 모든 run을 수집합니다. 특정 서브셋만 업로드하고 싶다면, 원하는 run들의 iterable을 직접 만들어서 importer에 전달할 수 있습니다.

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
Databricks MLFlow에서 가져오는 경우, 먼저 [Databricks CLI를 설정](https://docs.databricks.com/dev-tools/cli/index.html)해야 할 수 있습니다.

이전 단계에서 `mlflow-tracking-uri="databricks"`로 설정하세요.
{{% /alert %}}

artifacts 가져오기를 생략하려면, `artifacts=False`를 전달하세요:

```py
importer.import_runs(runs, artifacts=False)
```

특정 W&B Entity나 Project로 가져오고 싶다면, `Namespace`를 전달할 수 있습니다:

```py
from wandb.apis.importers import Namespace

importer.import_runs(runs, namespace=Namespace(entity, project))
```



## 데이터 내보내기

Public API를 사용하여 W&B에 저장한 데이터를 내보내거나 업데이트하세요. 이 API를 사용하기 전, 스크립트에서 데이터를 먼저 로깅해야 합니다. 자세한 내용은 [퀵스타트]({{< relref path="/guides/quickstart.md" lang="ko" >}})를 참고하세요.

**Public API 활용 유스 케이스**

- **데이터 내보내기**: 커스텀 분석을 위해 Jupyter 노트북으로 데이터프레임을 내려받을 수 있습니다. 데이터를 탐색한 후 분석 결과를 새로운 analysis run을 만들어 로그로 동기화할 수 있습니다. 예: `wandb.init(job_type="analysis")`
- **기존 Run 업데이트**: W&B run에 연관된 로그 데이터를 업데이트할 수 있습니다. 예를 들어, 여러 run의 설정을 업데이트하여 아키텍처 정보나 처음 로깅하지 않았던 하이퍼파라미터를 추가할 수 있습니다.

사용 가능한 함수는 [Generated Reference Docs]({{< relref path="/ref/python/public-api/" lang="ko" >}})에서 확인하세요.

### API 키 생성

API 키는 머신이 W&B에 인증할 때 사용합니다. 사용자 프로필에서 API 키를 생성할 수 있습니다.

{{% alert %}}
더 간편한 방법으로, [W&B 인증 페이지](https://wandb.ai/authorize)에 바로 접속해 API 키를 생성할 수 있습니다. 표시되는 API 키를 복사해 비밀번호 관리자와 같은 안전한 장소에 저장하세요.
{{% /alert %}}

1. 오른쪽 위 사용자 프로필 아이콘을 클릭합니다.
1. **User Settings**를 선택한 후, **API Keys** 섹션까지 스크롤합니다.
1. **Reveal**을 클릭하면 API 키가 표시됩니다. 복사하세요. 키를 숨기고 싶다면 페이지를 새로 고치세요.


### run path 찾기

Public API를 사용할 때는 `<entity>/<project>/<run_id>` 형식의 run path가 필요합니다. 앱 UI에서 run 페이지를 열고 [Overview 탭 ]({{< relref path="/guides/models/track/runs/#overview-tab" lang="ko" >}})을 클릭하면 run path를 확인할 수 있습니다.


### Run 데이터 내보내기

완료된 또는 활성화된 run에서 데이터를 다운로드하세요. 일반적으로 데이터프레임을 다운로드해 Jupyter 노트북에서 커스텀 분석을 하거나 자동화된 환경에서 커스텀 로직에 사용할 수 있습니다.

```python
import wandb

api = wandb.Api()
run = api.run("<entity>/<project>/<run_id>")
```

주로 사용되는 run 오브젝트의 속성은 다음과 같습니다:

| 속성            | 의미                                                                                                                                                                                                                                                                                                                  |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `run.config`    | 해당 run의 설정 정보를 담은 사전(dict), 예: 트레이닝 run의 하이퍼파라미터, 데이터셋 Artifact를 생성하는 run의 전처리 메소드 등. 이 정보들은 run의 입력값(input)이라 생각하면 됩니다.                                                                                                    |
| `run.history()` | 모델 트레이닝 중 변화하는 값(예: 손실값 등)을 저장하는 사전 리스트입니다. `run.log()`를 호출하면 이 객체에 값이 추가됩니다.                                                                                                                                                         |
| `run.summary`   | run의 결과를 요약해 저장하는 사전입니다. 일반적으로 정확도(accuracy), 손실(loss) 같은 스칼라나 대용량 파일도 포함할 수 있습니다. 기본적으로 `run.log()`는 summary에 마지막 시계열 값이 할당됩니다. summary 값은 직접 할당할 수도 있습니다. summary는 run의 출력값(output)으로 이해할 수 있습니다. |

과거 run의 데이터를 수정하거나 업데이트할 수도 있습니다. 기본적으로 하나의 api 객체 인스턴스가 모든 네트워크 요청을 캐싱합니다. 만약 실시간 정보가 필요한 경우, 실행 중인 스크립트에서 `api.flush()`를 호출해 값을 새로고침 하세요.

### 다양한 run 속성 이해하기

아래 예시는 run을 생성하고 데이터를 로깅한 후, run의 속성에 접근하는 방법을 보여줍니다:

```python
import wandb
import random

with wandb.init(project="public-api-example") as run:
    n_epochs = 5
    config = {"n_epochs": n_epochs}
    run.config.update(config)
    for n in range(run.config.get("n_epochs")):
        run.log(
            {"val": random.randint(0, 1000), "loss": (random.randint(0, 1000) / 1000.00)}
        )
```

이후 섹션에서 위 run 오브젝트 속성별로 결과가 어떻게 나오는지 설명합니다.

##### `run.config`

```python
{"n_epochs": 5}
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

기본 history 메소드는 메트릭을 고정된 샘플 수(기본값 500)로 샘플링합니다(`samples` 인수로 변경 가능). 대규모 run에서 모든 데이터를 내보내려면 `run.scan_history()` 메소드를 사용하세요. 자세한 내용은 [API Reference]({{< relref path="/ref/python/public-api" lang="ko" >}})를 참고하세요.

### 여러 Run 질의하기

{{< tabpane text=true >}}
    {{% tab header="DataFrame 및 CSV" %}}
아래 예제는 특정 프로젝트의 run을 찾아 이름, 설정, summary 통계까지 CSV로 출력하는 스크립트입니다. `<entity>`와 `<project>`는 W&B Entity명과 Project명으로 바꿔주세요.

```python
import pandas as pd
import wandb

api = wandb.Api()
entity, project = "<entity>", "<project>"
runs = api.runs(entity + "/" + project)

summary_list, config_list, name_list = [], [], []
for run in runs:
    # .summary는 accuracy 등 메트릭의 출력값을 포함합니다.
    #  ._json_dict로 대용량 파일은 제외시킵니다.
    summary_list.append(run.summary._json_dict)

    # .config는 하이퍼파라미터 등 설정값을 가집니다.
    #  _로 시작하는 특수값은 제외합니다.
    config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

    # .name은 사람이 읽기 쉬운 run 이름입니다.
    name_list.append(run.name)

runs_df = pd.DataFrame(
    {"summary": summary_list, "config": config_list, "name": name_list}
)

runs_df.to_csv("project.csv")

run.finish()
```
    {{% /tab %}}
    {{% tab header="MongoDB 스타일" %}}
W&B API는 프로젝트 내 여러 run을 api.runs()로 질의하는 방법도 제공합니다. 주로 run 데이터를 커스텀 분석 목적으로 내보낼 때 사용합니다. 쿼리 인터페이스는 [MongoDB 쿼리](https://docs.mongodb.com/manual/reference/operator/query)와 동일합니다.

```python
runs = api.runs(
    "username/project",
    {"$or": [{"config.experiment_name": "foo"}, {"config.experiment_name": "bar"}]},
)
print(f"Found {len(runs)} runs")
```    
    {{% /tab %}}
{{< /tabpane >}}

`api.runs`를 호출하면 반복 가능한 리스트 형태의 `Runs` 오브젝트가 반환됩니다. 기본적으로 한 번에 50개씩 순차적으로 불러오나, `per_page` 인수로 한번에 불러오는 개수를 변경할 수 있습니다.

`api.runs`는 `order` 인수도 받을 수 있습니다. 기본 정렬은 `-created_at`(최신순)이고, 오름차순은 `+created_at`입니다. 또한 config나 summary 값으로도 정렬이 가능합니다(예: `summary.val_acc`, `config.experiment_name` 등).

### 에러 처리

W&B 서버 통신에 실패할 경우 `wandb.CommError`가 발생합니다. 원래 예외는 `exc` 속성으로 확인할 수 있습니다.

### 최신 git 커밋 가져오기

UI에서 run을 클릭한 뒤 run 페이지의 Overview 탭에서 최신 git 커밋을 볼 수 있습니다. 혹은 `wandb-metadata.json` 파일에서도 확인 가능합니다. Public API에서는 `run.commit`으로 git 해시를 얻을 수 있습니다.

### 실행 중인 run의 이름 및 ID 가져오기

`wandb.init()` 호출 후, 스크립트 내에서 해당 run의 ID나 사람이 읽기 쉬운 이름에 바로 접근할 수 있습니다:

- 고유 run ID(8자 해시): `run.id`
- 무작위 run 이름(사람이 읽기 쉬움): `run.name`

run에 유용한 식별자를 설정하는 방법을 고민하고 있다면 다음 방식을 권장합니다:

- **Run ID**: 생성된 해시 그대로 두는 것을 권장. 프로젝트 내 run마다 고유해야 합니다.
- **Run name**: 짧고 읽기 쉽고, 가급적 유니크한 이름이면 차트에서 각각의 run을 구별하기 좋습니다.
- **Run notes**: run에서 하고 있는 작업을 간단히 설명하는데 적합합니다. `wandb.init(notes="your notes here")` 로 지정할 수 있습니다.
- **Run tags**: Run tag로 동적으로 추적하며, UI에서 필터 기능을 써서 원하는 run만 테이블에서 볼 수 있습니다. 스크립트에서 tag를 설정할 수 있고, run 페이지의 테이블이나 Overview 탭에서 나중에 수정할 수도 있습니다. 자세한 방법은 [여기]({{< relref path="/guides/models/track/runs/tags.md" lang="ko" >}})를 참고하세요.

## Public API 예시

### matplotlib 또는 seaborn용 데이터 내보내기

일반적인 데이터 내보내기 패턴은 [API 예시]({{< relref path="/ref/python/public-api/" lang="ko" >}})를 참고하세요. 커스텀 플롯이나 run 테이블에서 다운로드 버튼을 클릭해 브라우저에서 CSV로 바로 내릴 수도 있습니다.

### run에서 메트릭 읽기

아래 예제는 `run.log({"accuracy": acc})`로 저장된 시간(timestep) 및 정확도를 출력합니다. 대상 run은 `"<entity>/<project>/<run_id>"`입니다.

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
if run.state == "finished":
    for i, row in run.history().iterrows():
        print(row["_timestamp"], row["accuracy"])
```

### run 필터링

MongoDB 쿼리 언어를 사용해 필터링할 수 있습니다.

#### 날짜 기준

```python
runs = api.runs(
    "<entity>/<project>",
    {"$and": [{"created_at": {"$lt": "YYYY-MM-DDT##", "$gt": "YYYY-MM-DDT##"}}]},
)
```

### 특정 메트릭만 읽기

특정 메트릭만 추출하려면 `keys` 인수를 사용하세요. `run.history()`의 기본 샘플링 수는 500개입니다. 지정된 메트릭이 없는 스텝은 데이터프레임에서 `NaN`으로 표시됩니다. `keys`를 전달하면 선택한 메트릭을 가진 스텝을 더 자주 샘플링합니다.

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
if run.state == "finished":
    for i, row in run.history(keys=["accuracy"]).iterrows():
        print(row["_timestamp"], row["accuracy"])
```

### 두 run 비교

아래 코드는 `run1`과 `run2`의 설정 파라미터 차이를 출력합니다.

```python
import pandas as pd
import wandb

api = wandb.Api()

# <entity>, <project>, <run_id>를 변경하세요.
run1 = api.run("<entity>/<project>/<run_id>")
run2 = api.run("<entity>/<project>/<run_id>")


df = pd.DataFrame([run1.config, run2.config]).transpose()

df.columns = [run1.name, run2.name]
print(df[df[run1.name] != df[run2.name]])
```

출력 예시:

```
              c_10_sgd_0.025_0.01_long_switch base_adam_4_conv_2fc
batch_size                                 32                   16
n_conv_layers                               5                    4
optimizer                             rmsprop                 adam
```

### 진행이 끝난 run의 메트릭 업데이트

아래 예제는 이전 run의 정확도를 `0.9`로 수정하는 코드를 보여줍니다. 또한 이전 run의 정확도 히스토그램을 numpy 배열로 변경합니다.

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.summary["accuracy"] = 0.9
run.summary["accuracy_histogram"] = wandb.Histogram(numpy_array)
run.summary.update()
```

### 완료된 run의 메트릭 이름 변경

아래 코드는 summary 테이블의 기존 컬럼명을 새 이름으로 바꿉니다.

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.summary["new_name"] = run.summary["old_name"]
del run.summary["old_name"]
run.summary.update()
```

{{% alert %}}
컬럼명을 변경해도 테이블에만 반영됩니다. 차트에서는 여전히 기존 메트릭 이름을 사용합니다.
{{% /alert %}}



### 기존 run의 설정값(config) 업데이트

설정값 하나를 업데이트하는 예시입니다.

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.config["key"] = updated_value
run.update()
```

### 시스템 리소스 사용량을 CSV로 내보내기

아래 코드는 시스템 리소스 사용량 데이터를 찾고, 이를 CSV로 저장하는 예시입니다.

```python
import wandb

run = wandb.Api().run("<entity>/<project>/<run_id>")

system_metrics = run.history(stream="events")
system_metrics.to_csv("sys_metrics.csv")
```

### 샘플링되지 않은 메트릭 데이터 가져오기

기본적으로 history 데이터는 500개로 샘플링됩니다. 모든 로깅된 데이터를 원한다면 `run.scan_history()`로 불러올 수 있습니다. 아래는 기록된 모든 손실(loss)값을 다운로드하는 예시입니다.

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
history = run.scan_history()
losses = [row["loss"] for row in history]
```

### history에서 페이지 단위로 데이터 가져오기

백엔드에서 메트릭이 느리게 로드되거나 API 요청이 타임아웃되는 경우, `scan_history`의 page_size를 줄이면 개별 요청 타임아웃을 피할 수 있습니다. 기본값은 500이므로, 최적의 값을 찾아보세요.

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.scan_history(keys=sorted(cols), page_size=100)
```

### 프로젝트 내 모든 run의 메트릭을 CSV로 내보내기

아래 스크립트는 프로젝트의 run을 불러와 각 run의 이름, 설정, summary 통계를 포함한 데이터프레임과 CSV를 생성합니다. `<entity>`와 `<project>`는 적절한 값으로 변경하세요.

```python
import pandas as pd
import wandb

api = wandb.Api()
entity, project = "<entity>", "<project>"
runs = api.runs(entity + "/" + project)

summary_list, config_list, name_list = [], [], []
for run in runs:
    # .summary는 메트릭 등 출력값을 포함합니다.
    #  ._json_dict로 대용량 파일은 제외시킵니다.
    summary_list.append(run.summary._json_dict)

    # .config는 하이퍼파라미터 등 설정값을 가집니다.
    #  _로 시작하는 특수값은 제외합니다.
    config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

    # .name은 사람이 읽기 쉬운 run 이름입니다.
    name_list.append(run.name)

runs_df = pd.DataFrame(
    {"summary": summary_list, "config": config_list, "name": name_list}
)

runs_df.to_csv("project.csv")
```

### run의 시작 시간 얻기

아래 코드 조각은 run이 생성된 시점을 가져옵니다.

```python
import wandb

api = wandb.Api()

run = api.run("entity/project/run_id")
start_time = run.created_at
```

### 완료된 run에 파일 업로드

아래 예시는 완료된 run에 특정 파일을 업로드하는 코드입니다.

```python
import wandb

api = wandb.Api()

run = api.run("entity/project/run_id")
run.upload_file("file_name.extension")
```

### run에서 파일 다운로드

아래 코드는 run ID가 uxte44z7인 cifar 프로젝트에서 "model-best.h5" 파일을 찾아 로컬에 저장합니다.

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.file("model-best.h5").download()
```

### run의 모든 파일 다운로드

아래 예시는 run에 포함된 모든 파일을 찾고 로컬에 저장합니다.

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
for file in run.files():
    file.download()
```

### 특정 sweep의 run 가져오기

아래 스니펫은 지정 sweep에 연관된 모든 run을 다운로드합니다.

```python
import wandb

api = wandb.Api()

sweep = api.sweep("<entity>/<project>/<sweep_id>")
sweep_runs = sweep.runs
```

### sweep에서 가장 좋은 run 가져오기

아래 코드는 지정된 sweep에서 가장 좋은 run을 가져옵니다.

```python
import wandb

api = wandb.Api()

sweep = api.sweep("<entity>/<project>/<sweep_id>")
best_run = sweep.best_run()
```

`best_run`은 sweep 설정의 `metric` 파라미터에 정의된 가장 좋은 값의 run을 의미합니다.

### sweep에서 best model 파일 다운로드

아래 코드는 sweep 내 val accuracy가 가장 높은 run의 모델 파일(model.h5)을 다운로드합니다.

```python
import wandb

api = wandb.Api()

sweep = api.sweep("<entity>/<project>/<sweep_id>")
runs = sorted(sweep.runs, key=lambda run: run.summary.get("val_acc", 0), reverse=True)
val_acc = runs[0].summary.get("val_acc", 0)
print(f"Best run {runs[0].name} with {val_acc}% val accuracy")

runs[0].file("model.h5").download(replace=True)
print("Best model saved to model-best.h5")
```

### run에서 특정 확장자의 파일 모두 삭제

아래 코드는 run 내에서 지정된 확장자를 가진 파일을 모두 삭제합니다.

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

아래 코드는 run의 모든 시스템 자원 사용량 메트릭을 포함한 데이터프레임을 생성하고 CSV로 저장합니다.

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
system_metrics = run.history(stream="events")
system_metrics.to_csv("sys_metrics.csv")
```

### 요약 메트릭 업데이트

사전을 전달하여 summary 메트릭을 업데이트할 수 있습니다.

```python
summary.update({"key": val})
```

### run에서 실행된 명령어 추출

각 run에서는 run overview 페이지에 해당 run을 시작한 명령어도 기록합니다. 이 명령어를 Public API를 사용해 직접 가져오려면 아래와 같이 할 수 있습니다:

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")

meta = json.load(run.file("wandb-metadata.json").download())
program = ["python"] + [meta["program"]] + meta["args"]
```
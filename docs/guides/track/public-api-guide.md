---
description: Import data from MLFlow, export or update data that you have saved to
  W&B
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 데이터 가져오기 및 내보내기

<head>
  <title>W&B로 데이터 가져오기 및 내보내기</title>
</head>

MLFlow 또는 W&B 인스턴스 간에 데이터를 가져오거나 내보낼 수 있습니다.

## MLFlow에서 데이터 가져오기

W&B는 실험, 실행, 아티팩트, 메트릭 및 기타 메타데이터를 포함하여 MLFlow에서 데이터를 가져오는 것을 지원합니다.

의존성 설치:

```shell
pip install mlflow wandb>=0.14.0
```

W&B 로그인 (이전에 로그인한 적이 없다면 프롬프트가 표시됩니다)

```shell
wandb login
```

기존 MLFlow 서버에서 모든 실행 가져오기:

```shell
wandb import mlflow \ &&
    --mlflow-tracking-uri <mlflow_uri> \ &&
    --target-entity       <entity> \ &&
    --target-project      <project>
```

:::tip
Databricks MLFlow에서 가져올 경우 먼저 [Databricks CLI를 구성](https://docs.databricks.com/dev-tools/cli/index.html)해야 할 수 있습니다.

이전 단계에서 `--mlflow-tracking-uri=databricks`를 설정하세요.
:::

#### 고급

Python에서도 가져올 수 있습니다. 오버라이드를 지정하거나 명령줄보다 Python을 선호하는 경우 유용할 수 있습니다.

```py
from wandb.apis.importers import MlflowImporter

# 모든 가져온 실행에 대한 설정을 오버라이드하는 선택적 사전
overrides = {"entity": "my_custom_entity", "project": "my_custom_project"}

importer = MlflowImporter(mlflow_tracking_uri="...")
importer.import_all_parallel()
```

더 세밀한 제어를 위해, 실험을 선택적으로 가져오거나 자체 사용자 정의 로직에 기반하여 오버라이드를 지정할 수 있습니다. 예를 들어, 다음 코드는 사용자 지정 태그가 있는 실행을 만든 다음 지정된 프로젝트로 가져오는 방법을 보여줍니다.

```py
default_settings = {"entity": "default_entity", "project": "default_project"}

special_tag_settings = {"entity": "special_entity", "project": "special_project"}

for run in importer.download_all_runs():
    if "special_tag" in run.tags():
        overrides = special_tag_settings
    else:
        overrides = default_settings

    importer.import_run(run, overrides=overrides)
```

## 다른 W&B 인스턴스에서 데이터 가져오기

:::info
이 기능은 베타 버전이며 W&B 공용 클라우드에서만 가져오기를 지원합니다.
:::

의존성 설치:

```sh
pip install wandb>=0.15.6 polars tqdm
```

W&B 로그인. 이전에 로그인하지 않았다면 프롬프트를 따르세요.

```sh
wandb login
```

Python에서 가져오기 도구 인스턴스화:

```
from wandb.apis.importers import WandbParquetImporter

importer = WandbParquetImporter(
    src_base_url="https://api.wandb.ai",
    src_api_key="your-api-key-here",
    dst_base_url="https://example-target.wandb.io",
    dst_api_key="target-environment-api-key-here",
)
```

### 실행 가져오기

엔터티에서 모든 W&B 실행 가져오기:

```py
importer.import_all_runs(src_entity)
```

기본적으로 모든 프로젝트를 가져오지 않으려면 프로젝트를 선택적으로 지정할 수 있습니다:

```py
importer.import_all_runs(src_entity, src_project)
```

데이터를 다른 엔터티 또는 프로젝트로 가져오려면 `overrides`로 지정할 수 있습니다:

```py
importer.import_all_runs(
    src_entity, src_project, overrides={"entity": dst_entity, "project": dst_project}
)
```

### 리포트 가져오기

엔터티에서 모든 리포트 가져오기:

```py
importer.import_all_reports(src_entity)
```

기본적으로 모든 프로젝트를 가져오지 않으려면 프로젝트를 선택적으로 지정할 수 있습니다:

```py
importer.import_all_reports(src_entity, src_project)
```

데이터를 다른 엔터티 또는 프로젝트로 가져오려면 `overrides` 매개변수를 지정하세요. 리포트 오버라이드는 다른 이름 및 설명도 지원합니다:

```py
importer.import_all_reports(
    src_entity, src_project, overrides={"entity": dst_entity, "project": dst_project}
)
```

### 개별 실행 및 리포트 가져오기

가져오기 도구는 가져오기를 더 세밀하게 제어할 수 있도록 지원합니다.

`import_run` 및 `import_report`를 각각 사용하여 개별 실행 및 리포트를 가져올 수 있습니다.

### 사용자 정의 로직을 사용하여 실행 및 리포트 가져오기

자체 사용자 정의 로직을 기반으로 실행 및 리포트 목록을 수집하고 가져올 수 있습니다. 예를 들어:

```py
runs = importer.collect_runs(src_entity)

for run in runs:
    if run.name().startswith("something-important"):
        importer.import_run(run)
```

## 데이터 내보내기

Public API를 사용하여 W&B에 저장한 데이터를 내보내거나 업데이트합니다. 이 API를 사용하기 전에 스크립트에서 데이터를 로깅하고 싶을 것입니다 — 자세한 내용은 [퀵스타트](../../quickstart.md)를 확인하세요.

**Public API 사용 사례**

- **데이터 내보내기**: 사용자 지정 분석을 위해 데이터프레임을 내려받아 Jupyter 노트북에서 탐색합니다. 데이터를 탐색한 후 새로운 분석 실행을 생성하고 결과를 로깅하여 발견한 내용을 동기화할 수 있습니다. 예: `wandb.init(job_type="analysis")`
- **기존 실행 업데이트**: W&B 실행과 연결된 데이터를 업데이트할 수 있습니다. 예를 들어, 아키텍처 또는 원래 로깅되지 않은 하이퍼파라미터와 같은 추가 정보를 포함하도록 일련의 실행의 구성을 업데이트하고 싶을 수 있습니다.

사용 가능한 함수에 대한 자세한 내용은 [생성된 참조 문서](../../ref/python/public-api/README.md)를 참조하세요.

### 인증

다음 두 가지 방법 중 하나로 [API 키](https://wandb.ai/authorize)를 사용하여 기계를 인증하세요:

1. 명령줄에서 `wandb login`을 실행하고 API 키를 붙여넣습니다.
2. `WANDB_API_KEY` 환경 변수를 API 키로 설정합니다.

### 실행 경로 찾기

Public API를 사용하려면 종종 실행 경로인 `<entity>/<project>/<run_id>`가 필요합니다. 앱 UI에서 실행 페이지를 열고 [Overview 탭](../app/pages/run-page.md#overview-tab)을 클릭하여 실행 경로를 얻습니다.

### 실행 데이터 내보내기

완료되었거나 활성 상태인 실행에서 데이터를 다운로드합니다. 일반적인 사용 사례에는 Jupyter 노트북에서 사용자 지정 분석을 위해 데이터프레임을 다운로드하거나 자동화된 환경에서 사용자 정의 로직을 사용하는 것이 포함됩니다.

```python
import wandb

api = wandb.Api()
run = api.run("<entity>/<project>/<run_id>")
```

실행 객체의 가장 일반적으로 사용되는 속성은 다음과 같습니다:

| 속성             | 의미                                                                                                                                                                                                                                                   |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `run.config`    | 데이터세트 아티팩트를 생성하는 실행에 대한 전처리 방법 또는 학습 실행에 대한 하이퍼파라미터와 같은 실행의 구성 정보의 사전입니다. 이는 실행의 "입력"으로 생각할 수 있습니다.                                                                                                     |
| `run.history()` | 모델이 학습하는 동안 변경되는 값, 예를 들어 손실을 저장하기 위한 사전 목록입니다. `wandb.log()` 명령은 이 객체에 추가합니다.                                                                                                                                                      |
| `run.summary`   | 실행 결과를 요약하는 정보의 사전입니다. 이는 정확도 및 손실과 같은 스칼라 또는 큰 파일일 수 있습니다. 기본적으로 `wandb.log()`는 로깅된 시계열의 최종 값으로 요약을 설정합니다. 요약의 내용은 직접 설정할 수도 있습니다. 요약을 실행의 "출력"으로 생각하십시오. |

과거 실행의 데이터를 수정하거나 업데이트할 수도 있습니다. 기본적으로 api 객체의 단일 인스턴스는 모든 네트워크 요청을 캐시합니다. 실행 스크립트에서 실시간 정보가 필요한 사용 사례의 경우 `api.flush()`를 호출하여 업데이트된 값을 얻습니다.

### 다른 속성 이해하기

아래 실행의 경우

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

위 실행 객체 속성에 대한 다른 출력입니다

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

기본적으로 history 메서드는 메트릭을 고정된 샘플 수(기본값은 500, `samples` 인수로 변경할 수 있음)로 샘플링합니다. 큰 실행에서 모든 데이터를 내보내려면 `run.scan_history()` 메서드를 사용할 수 있습니다. 자세한 내용은 [API 참조](https://docs.wandb.ai/ref/python/public-api)를 참조하세요.

### 여러 실행 쿼리하기

<Tabs
defaultValue="dataframes_csvs"
values={[
{label: '데이터프레임 및 CSV', value: 'dataframes_csvs'},
{label: 'MongoDB 스타일', value: 'mongoDB'},
]}>
<TabItem value="dataframes_csvs">

이 예제 스크립트는 프로젝트를 찾고 이름, 구성 및 요약 통계가 포함된 실행의 CSV를 출력합니다. `<entity>` 및 `<project>`를 각각 W&B 엔터티 및 프로젝트 이름으로 바꾸세요.

```python
import pandas as pd
import wandb

api = wandb.Api()
entity, project = "<entity>", "<project>"
runs = api.runs(entity + "/" + project)

summary_list, config_list, name_list = [], [], []
for run in runs:
    # .summary는 정확도와 같은 메트릭에 대한 출력 키/값을 포함합니다.
    #  큰 파일을 생략하기 위해 ._json_dict를 호출합니다
    summary_list.append(run.summary._json_dict)

    # .config는 하이퍼파라미터와 같은 구성을 포함합니다.
    #  _로 시작하는 특수 값을 제거합니다.
    config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

    # .name은 실행의 사람이 읽을 수 있는 이름입니다.
    name_list.append(run.name)

runs_df = pd.DataFrame(
    {"summary": summary_list, "config": config_list, "name": name_list}
)

runs_df.to_csv("project.csv")
```

  </TabItem>
  <TabItem value="mongoDB">

W&B API는 api.runs()를 사용하여 프로젝트에서 실행을 쿼리할 수 있는 방법도 제공합니다. 가장 일반적인 사용 사례는 사용자 지정 분석을 위해 실행 데이터를 내보내는 것입니다. 쿼리 인터페이스는 [MongoDB가 사용하는](https://docs.mongodb.com/manual/reference/operator/query) 것과 동일합니다.

```python
runs = api.runs(
    "username/project",
    {"$or": [{"config.experiment_name": "foo"}, {"config.experiment_name": "bar"}]},
)
print(f"Found {len(runs)} runs")
```

  </TabItem>
</Tabs>

`api.runs`를 호출하면 반복 가능하고 리스트처럼 작동하는 `Runs` 객체를 반환합니다. 기본적으로 객체는 필요에 따라 한 번에 50개의 실행을 순차적으로 로드하지만, `per_page` 키워드 인수로 로드되는 페이지 수를 변경할 수 있습니다.

`api.runs`는 또한 `order` 키워드 인수를 허용합니다. 기본 순서는 `-created_at`이며, 오름차순으로 결과를 얻으려면 `+created_at`을 지정하세요. `summary.val_acc` 또는 `config.experiment_name`과 같은 구성 또는 요약 값으로도 정렬할 수 있습니다.

### 오류 처리

W&B 서버와 통신하는 동안 오류가 발생하면 `wandb.CommError`가 발생합니다. 원래 예외는 `exc` 속성을 통해 조사할 수 있습니다.

### API를 통해 최신 git 커밋 가져오기

UI에서 실행을 클릭한 다음 실행 페이지에서 Overview 탭을 클릭하면 최신 git 커밋을 볼 수 있습니다. 파일 `wandb-metadata.json`에도 있습니다. public API를 사용하여 `run.commit`으로 git 해시를 가져올 수 있습니다.

## 자주 묻는 질문

### matplotlib 또는 seaborn에서 시각화하기 위해 데이터를 내보내는 방법은 무엇입니까?

일반적인 내보내기 패턴에 대한 몇 가지 [API 예제](../../ref/python/public-api/README.md)를 확인하세요. 사용자 지정 플롯 또는 확장된 실행 테이블에서 다운로드 버튼을 클릭하여 브라우저에서 CSV를 다운로드할 수도 있습니다.

### 실행 중에 실행의 이름과 ID를 얻는 방법은 무엇입니까?

`wandb.init()`을 호출한 후 스크립트에서 랜덤 실행 ID 또는 사람이 읽을 수 있는 실행 이름에 접근할 수 있습니다:

- 고유 실행 ID (8자리 해시): `wandb.run.id`
- 랜덤 실행 이름 (사람이 읽을 수 있음): `wandb.run.name`

실행에 유용한 식별자를 설정하는 방법을 고민하고 있다면, 다음을 권장합니다:

- **실행 ID**: 생성된 해시로

### 모든 실행에서 메트릭을 CSV 파일로 내보내기

이 스크립트는 프로젝트의 실행을 가져와서 그 이름, 설정, 요약 통계를 포함하는 데이터프레임과 CSV를 생성합니다. `<entity>`와 `<project>`를 여러분의 W&B 엔티티와 프로젝트 이름으로 각각 대체하세요.

```python
import pandas as pd
import wandb

api = wandb.Api()
entity, project = "<entity>", "<project>"
runs = api.runs(entity + "/" + project)

summary_list, config_list, name_list = [], [], []
for run in runs:
    # .summary에는 정확도와 같은 메트릭에 대한 출력 키/값이 포함됩니다.
    #  큰 파일을 생략하기 위해 ._json_dict를 호출합니다
    summary_list.append(run.summary._json_dict)

    # .config에는 하이퍼파라미터가 포함됩니다.
    #  _로 시작하는 특수 값을 제거합니다.
    config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

    # .name은 실행의 인간 친화적 이름입니다.
    name_list.append(run.name)

runs_df = pd.DataFrame(
    {"summary": summary_list, "config": config_list, "name": name_list}
)

runs_df.to_csv("project.csv")
```

### 실행의 시작 시간 가져오기

이 코드 조각은 실행이 생성된 시간을 검색합니다.

```python
import wandb

api = wandb.Api()

run = api.run("entity/project/run_id")
start_time = run.created_at
```

### 완료된 실행에 파일 업로드하기

아래 코드 조각은 선택된 파일을 완료된 실행에 업로드합니다.

```python
import wandb

api = wandb.Api()

run = api.run("entity/project/run_id")
run.upload_file("file_name.extension")
```

### 실행에서 파일 다운로드하기

이것은 cifar 프로젝트의 실행 ID uxte44z7과 연관된 "model-best.h5" 파일을 찾아 로컬에 저장합니다.

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.file("model-best.h5").download()
```

### 실행에서 모든 파일 다운로드하기

이것은 실행과 관련된 모든 파일을 찾아 로컬에 저장합니다.

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
for file in run.files():
    file.download()
```

### 특정 스윕에서 실행 가져오기

이 조각은 특정 스윕과 연관된 모든 실행을 다운로드합니다.

```python
import wandb

api = wandb.Api()

sweep = api.sweep("<entity>/<project>/<sweep_id>")
sweep_runs = sweep.runs
```

### 스윕에서 최고의 실행 가져오기

다음 조각은 주어진 스윕에서 최고의 실행을 가져옵니다.

```python
import wandb

api = wandb.Api()

sweep = api.sweep("<entity>/<project>/<sweep_id>")
best_run = sweep.best_run()
```

`best_run`은 스윕 설정의 `metric` 파라미터에 의해 정의된 최고의 메트릭을 가진 실행입니다.

### 스윕에서 최고의 모델 파일 다운로드하기

이 조각은 `model.h5`로 모델 파일을 저장한 실행이 포함된 스윕에서 검증 정확도가 가장 높은 모델 파일을 다운로드합니다.

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

### 실행에서 주어진 확장자를 가진 모든 파일 삭제하기

이 조각은 실행에서 주어진 확장자를 가진 파일들을 삭제합니다.

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

### 시스템 메트릭 데이터 다운로드하기

이 조각은 실행에 대한 모든 시스템 자원 소비 메트릭을 포함하는 데이터프레임을 생성한 다음 CSV로 저장합니다.

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
system_metrics = run.history(stream="events")
system_metrics.to_csv("sys_metrics.csv")
```

### 요약 메트릭 업데이트하기

요약 메트릭을 업데이트하기 위해 사전을 전달할 수 있습니다.

```python
summary.update({"key": val})
```

### 실행을 시작한 명령 가져오기

각 실행은 실행 개요 페이지에서 실행을 시작한 명령을 캡처합니다. 이 명령을 API에서 다운로드하려면 다음을 실행할 수 있습니다:

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")

meta = json.load(run.file("wandb-metadata.json").download())
program = ["python"] + [meta["program"]] + meta["args"]
```
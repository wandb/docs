---
title: Import and export data
description: MLFlow에서 데이터를 가져오거나, W&B에 저장한 데이터를 내보내거나 업데이트합니다.
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

W&B 공개 API를 사용하여 MLFlow 또는 W&B 인스턴스 간에 데이터를 내보내거나 가져오세요.

:::info
이 기능은 python>=3.8이 필요합니다.
:::

## MLFlow에서 데이터 가져오기

W&B는 MLFlow에서 실험, runs, artifacts, 메트릭 및 기타 메타데이터를 가져오는 것을 지원합니다.

의존성 설치:

```shell
# note: this requires py38+
pip install wandb[importers]
```

W&B에 로그인합니다. 처음 로그인하는 경우 화면의 지시를 따르세요.

```shell
wandb login
```

기존 MLFlow 서버에서 모든 runs를 가져옵니다:

```py
from wandb.apis.importers.mlflow import MlflowImporter

importer = MlflowImporter(mlflow_tracking_uri="...")

runs = importer.collect_runs()
importer.import_runs(runs)
```

기본적으로 `importer.collect_runs()`는 MLFlow 서버에서 모든 runs를 수집합니다. 특정 서브셋만 업로드하고 싶다면, 스스로 runs 이터러블을 구성하여 가져오기 도구에 전달할 수 있습니다.

```py
import mlflow
from wandb.apis.importers.mlflow import MlflowRun

client = mlflow.tracking.MlflowClient(mlflow_tracking_uri)

runs: Iterable[MlflowRun] = []
for run in mlflow_client.search_runs(...):
    runs.append(MlflowRun(run, client))

importer.import_runs(runs)
```

:::tip
Databricks MLFlow에서 가져오려면 [Databricks CLI를 먼저 구성해야 할 수도 있습니다](https://docs.databricks.com/dev-tools/cli/index.html).

`mlflow-tracking-uri="databricks"`를 이전 단계에서 설정하세요.
:::

artifacts 가져오기를 생략하려면 `artifacts=False`를 전달하세요:

```py
importer.import_runs(runs, artifacts=False)
```

특정 W&B 엔터티와 프로젝트에 가져오려면 `Namespace`를 전달할 수 있습니다:

```py
from wandb.apis.importers import Namespace

importer.import_runs(runs, namespace=Namespace(entity, project))
```

## 다른 W&B 인스턴스에서 데이터 가져오기

:::info
이 기능은 베타 상태이며, W&B 공개 클라우드에서만 가져오기를 지원합니다.
:::

의존성 설치:

```sh
# note: this requires py38+
pip install wandb[importers]
```

원본 W&B 서버에 로그인합니다. 처음 로그인하는 경우 화면의 지시를 따르세요.

```sh
wandb login
```

원본 W&B 인스턴스에서 대상 W&B 인스턴스로 모든 runs와 artifacts를 가져옵니다. Runs와 artifacts는 대상 인스턴스의 해당 네임스페이스에 가져옵니다.

```py
from wandb.apis.importers.wandb import WandbImporter
from wandb.apis.importers import Namespace

importer = WandbImporter(
    src_base_url="https://api.wandb.ai",
    src_api_key="your-api-key-here",
    dst_base_url="https://example-target.wandb.io",
    dst_api_key="target-environment-api-key-here",
)

# "entity/project"에서 "entity/project"로 src의 모든 runs, artifacts, reports를 가져옵니다
importer.import_all(namespaces=[
    Namespace(entity, project),
    # ... 여기에 더 많은 네임스페이스를 추가하세요
])
```

대상 네임스페이스를 변경하고 싶다면 `remapping: dict[Namespace, Namespace]`를 지정할 수 있습니다.

```py
importer.import_all(
    namespaces=[Namespace(entity, project)],
    remapping={
        Namespace(entity, project): Namespace(new_entity, new_project),
    }
)
```

기본적으로 import는 증분형입니다. 이후의 import는 이전 작업을 검증하고 성공/실패를 추적하는 `.jsonl` 파일에 작성하려 시도합니다. import가 성공하면 이후의 검증은 건너뛰어집니다. import가 실패하면 재시도합니다. 이를 비활성화하려면 `incremental=False`를 설정하세요.

```py
importer.import_all(
    namespaces=[Namespace(entity, project)],
    incremental=False,
)
```

### 알려진 문제 및 제약 사항

- 대상 네임스페이스가 존재하지 않으면 W&B가 자동으로 생성합니다.
- 대상 네임스페이스에서 run이나 artifact에 동일한 ID가 있으면 W&B는 이를 증분 import로 처리합니다. 대상 run/artifact는 이전 import에서 실패한 경우 검증하고 재시도합니다.
- 원본 시스템에서 데이터는 절대 삭제되지 않습니다.

1. 때때로 대량 import할 때(특히 큰 artifacts), S3 속도 제한에 걸릴 수 있습니다. `botocore.exceptions.ClientError: An error occurred (SlowDown) when calling the PutObject operation` 오류가 발생하면 몇 개의 네임스페이스만 한 번에 이동하여 간격을 두고 import를 시도할 수 있습니다.
2. 작업 공간에서 가져온 run 테이블은 비어 있을 수 있지만, Artifacts 탭으로 이동하여 해당 run 테이블 아티팩트를 클릭하면 예상대로 테이블을 볼 수 있습니다.
3. wandb.log를 사용하여 명시적으로 기록되지 않은 시스템 메트릭 및 사용자 정의 차트는 가져오지 않습니다.

## 데이터 내보내기

W&B에 저장한 데이터를 내보내거나 업데이트하려면 공개 API를 사용하십시오. 이 API를 사용하기 전에 스크립트에서 데이터를 기록해야 합니다 - 자세한 내용은 [퀵스타트](../../quickstart.md)를 확인하세요.

**공개 API의 유스 케이스**

- **데이터 내보내기**: Jupyter 노트북에서 사용자 정의 분석을 위해 데이터 프레임을 끌어옵니다. 데이터를 탐색한 후, 새로운 분석 run을 생성하고 결과를 기록하여 발견한 내용을 동기화할 수 있습니다. 예를 들어: `wandb.init(job_type="analysis")`
- **기존 Runs 업데이트**: W&B run과 관련된 기록 데이터를 업데이트할 수 있습니다. 예를 들어, 처음에는 기록되지 않았던 아키텍처 또는 하이퍼파라미터와 같은 추가 정보를 포함하도록 여러 run의 구성을 업데이트할 수 있습니다.

사용 가능한 함수에 대한 자세한 내용은 [생성된 참조 문서](../../ref/python/public-api/README.md)를 참조하세요.

### 인증

두 가지 방법 중 하나로 [API 키](https://wandb.ai/authorize)를 사용하여 머신을 인증하십시오:

1. 커맨드 라인에서 `wandb login`을 실행하고 API 키를 붙여넣습니다.
2. `WANDB_API_KEY` 환경 변수를 API 키로 설정합니다.

### run 경로 찾기

공개 API를 사용하려면 종종 `<entity>/<project>/<run_id>` 형식의 run 경로가 필요합니다. 앱 UI에서 run 페이지를 열고 [Overview 탭](../app/pages/run-page.md#overview-tab)을 클릭하여 run 경로를 얻으십시오.

### Run 데이터 내보내기

종료된 또는 활성화된 run에서 데이터를 다운로드합니다. 일반적인 사용 사례로는 Jupyter 노트북에서 사용자 정의 분석을 위해 데이터 프레임을 다운로드하거나 자동화된 환경에서 사용자 정의 로직을 사용하는 것이 포함됩니다.

```python
import wandb

api = wandb.Api()
run = api.run("<entity>/<project>/<run_id>")
```

run 오브젝트의 가장 일반적으로 사용되는 속성은 다음과 같습니다:

| 속성            | 의미                                                                                                                                                                                                                            |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `run.config`    | 트레이닝 run의 하이퍼파라미터 또는 데이터셋 Artifact를 생성하는 run의 전처리 메소드를 포함한 run의 설정 정보 사전입니다. 이를 run의 "입력"으로 생각하세요.                                                                                                    |
| `run.history()` | 모델이 트레이닝되는 동안 변화하는 값들을 저장하기 위한 사전 목록입니다. 명령 `wandb.log()`는 이 오브젝트에 추가됩니다.                                                                                                                                                                 |
| `run.summary`   | run의 결과를 요약한 정보의 사전입니다. 이것은 정확도와 손실과 같은 스칼라일 수도 있고, 큰 파일일 수도 있습니다. 기본적으로 `wandb.log()`는 summary를 기록된 시계열의 최종 값으로 설정합니다. summary의 내용은 직접 설정할 수도 있습니다. summary를 run의 "출력"으로 생각하세요. |

과거 run의 데이터를 수정하거나 업데이트할 수도 있습니다. 기본적으로 api 오브젝트의 단일 인스턴스는 모든 네트워크 요청을 캐싱합니다. 사용 사례가 실행 중인 스크립트에서 실시간 정보가 필요한 경우 `api.flush()`를 호출하여 업데이트된 값을 얻으십시오.

### 다양한 속성 이해하기

아래 run의 경우

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

위의 run 오브젝트 속성에 대한 다양한 출력은 다음과 같습니다.

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

기본 history 메소드는 메트릭을 고정된 수의 샘플로 샘플링합니다(기본값은 500이며, `samples` 인수로 변경할 수 있습니다). 큰 run에서 모든 데이터를 내보내고자 할 때 `run.scan_history()` 메소드를 사용할 수 있습니다. 자세한 내용은 [API 참조](/ref/python/public-api)를 참조하세요.

### 여러 Runs 쿼리

<Tabs
defaultValue="dataframes_csvs"
values={[
{label: 'Dataframes and CSVs', value: 'dataframes_csvs'},
{label: 'MongoDB Style', value: 'mongoDB'},
]}>
<TabItem value="dataframes_csvs">

이 예제 스크립트는 프로젝트를 찾고 이름, 구성 및 요약 통계가 포함된 runs의 CSV를 출력합니다. `<entity>`와 `<project>`를 사용자의 W&B 엔터티와 프로젝트 이름으로 각각 바꾸세요.

```python
import pandas as pd
import wandb

api = wandb.Api()
entity, project = "<entity>", "<project>"
runs = api.runs(entity + "/" + project)

summary_list, config_list, name_list = [], [], []
for run in runs:
    # .summary는 정확도와 같은 메트릭에 대한 출력 키/값을 포함합니다.
    #  ._json_dict를 호출하여 큰 파일을 생략합니다.
    summary_list.append(run.summary._json_dict)

    # .config는 하이퍼파라미터를 포함합니다.
    #  _로 시작하는 특수 값을 제거합니다.
    config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

    # .name은 run의 사람이 읽을 수 있는 이름입니다.
    name_list.append(run.name)

runs_df = pd.DataFrame(
    {"summary": summary_list, "config": config_list, "name": name_list}
)

runs_df.to_csv("project.csv")
```

</TabItem>
<TabItem value="mongoDB">

W&B API는 또한 api.runs()를 사용하여 프로젝트에서 runs를 쿼리할 수 있는 방법을 제공합니다. 가장 일반적인 사용 사례는 사용자 정의 분석을 위해 runs 데이터를 내보내는 것입니다. 쿼리 인터페이스는 [MongoDB에서 사용하는 것](https://docs.mongodb.com/manual/reference/operator/query)과 동일합니다.

```python
runs = api.runs(
    "username/project",
    {"$or": [{"config.experiment_name": "foo"}, {"config.experiment_name": "bar"}]},
)
print(f"Found {len(runs)} runs")
```

</TabItem>
</Tabs>

`api.runs`를 호출하면 리스트처럼 작동하는 반복 가능한 `Runs` 오브젝트가 반환됩니다. 기본적으로 오브젝트는 필요에 따라 한 번에 50개의 run을 순서대로 로드하지만 `per_page` 키워드 인수로 한 페이지에 로드되는 수를 변경할 수 있습니다.

`api.runs`는 또한 `order` 키워드 인수를 수락합니다. 기본 순서는 `-created_at`이며, 결과를 오름차순으로 얻으려면 `+created_at`을 지정하세요. config 또는 summary 값에 따라 정렬할 수도 있습니다. 예: `summary.val_acc` 또는 `config.experiment_name`

### 오류 처리

W&B 서버와의 통신 중 오류가 발생하면 `wandb.CommError`가 발생합니다. 원래 예외는 `exc` 속성을 통해 확인할 수 있습니다.

### API를 통해 최신 git 커밋 가져오기

UI에서 run을 클릭한 후 run 페이지의 Overview 탭을 클릭하여 최신 git 커밋을 확인할 수 있습니다. 이는 또한 `wandb-metadata.json` 파일에 있습니다. 공개 API를 사용하여 git 해시를 `run.commit`로 가져올 수 있습니다.

## 자주 묻는 질문

### matplotlib 또는 seaborn에서 시각화를 위해 데이터를 내보내려면 어떻게 해야 하나요?

일반적인 내보내기 패턴에 대한 몇 가지 예를 보려면 [API 예제](../../ref/python/public-api/README.md)를 확인하세요. 맞춤형 플롯이나 확장된 runs 테이블에서 다운로드 버튼을 클릭하여 브라우저에서 CSV를 다운로드할 수도 있습니다.

### run 중에 run의 이름과 ID를 얻으려면 어떻게 해야 하나요?

`wandb.init()` 호출 후 스크립트에서 다음과 같이 고유한 run ID 또는 사람이 읽을 수 있는 run 이름에 엑세스할 수 있습니다.

- 고유한 run ID (8자 해시): `wandb.run.id`
- 무작위 run 이름 (사람이 읽을 수 있는 형태): `wandb.run.name`

실행에 유용한 식별자를 설정하는 방법을 고민하고 있다면 다음을 권장합니다:

- **Run ID**: 생성된 해시로 그대로 두세요. 이는 프로젝트 내 다른 run들 사이에서 유일해야 합니다.
- **Run 이름**: 복잡하지 않고 읽을 수 있으며, 서로 다른 차트의 선을 구별할 수 있도록 되도록 유일해야 합니다.
- **Run 노트**: 실행 중 하고 있는 일을 간단히 설명하기 좋은 위치입니다. `wandb.init(notes="your notes here")`로 설정할 수 있습니다.
- **Run 태그**: 실행 태그에서 동적으로 추적하고, UI에서 필터를 사용하여 관심 있는 runs로 테이블을 필터링할 수 있습니다. 스크립트에서 태그를 설정한 다음, UI에서도 편집할 수 있으며, 이는 runs 테이블 및 run 페이지의 Overview 탭에서 모두 가능합니다. 자세한 지침은 [여기](../app/features/tags.md)를 참조하세요.

## 공개 API 예제

### run에서 메트릭 읽기

이 예제는 run에 저장된 `wandb.log({"accuracy": acc})`와 함께 저장된 타임스탬프와 정확성을 출력합니다.

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
if run.state == "finished":
    for i, row in run.history().iterrows():
        print(row["_timestamp"], row["accuracy"])
```

### Runs 필터링

MongoDB 쿼리 언어를 사용하여 필터를 사용할 수 있습니다.

#### 날짜

```python
runs = api.runs(
    "<entity>/<project>",
    {"$and": [{"created_at": {"$lt": "YYYY-MM-DDT##", "$gt": "YYYY-MM-DDT##"}}]},
)
```

### 특정 메트릭을 run에서 읽기

run에서 특정 메트릭을 가져오려면 `keys` 인수를 사용하세요. `run.history()`를 사용할 때 기본 샘플 수는 500입니다. 특정 메트릭을 포함하지 않는 기록된 단계는 출력 데이터 프레임에서 `NaN`으로 표시됩니다. `keys` 인수는 나열된 메트릭 키를 포함하는 단계가 더 자주 샘플링되도록 API에 지시합니다.

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
if run.state == "finished":
    for i, row in run.history(keys=["accuracy"]).iterrows():
        print(row["_timestamp"], row["accuracy"])
```

### 두 run을 비교

이 코드는 `run1`과 `run2` 사이의 다름을 가지는 구성 파라미터를 출력합니다.

```python
import pandas as pd
import wandb

api = wandb.Api()

# replace with your <entity>, <project>, and <run_id>
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

### run이 끝난 후, run의 메트릭 업데이트

이 예제는 이전 run의 정확도를 `0.9`로 설정합니다. 또한 이전 run의 정확도 히스토그램을 `numpy_array`의 히스토그램으로 수정합니다.

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.summary["accuracy"] = 0.9
run.summary["accuracy_histogram"] = wandb.Histogram(numpy_array)
run.summary.update()
```

### run이 끝난 후, run의 메트릭 이름 변경

이 예제는 테이블에서 summary 컬럼의 이름을 변경합니다.

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.summary["new_name"] = run.summary["old_name"]
del run.summary["old_name"]
run.summary.update()
```

:::caution
컬럼 이름 변경은 테이블에만 적용됩니다. 차트는 여전히 원래 이름으로 메트릭을 참조합니다.
:::

### 기존 run의 설정 업데이트

이 예제는 설정 중 하나를 업데이트합니다.

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.config["key"] = updated_value
run.update()
```

### 시스템 리소스 사용량을 CSV 파일로 내보내기

아래 코드조각은 시스템 리소스 사용량을 찾아 CSV에 저장합니다.

```python
import wandb

run = wandb.Api().run("<entity>/<project>/<run_id>")

system_metrics = run.history(stream="events")
system_metrics.to_csv("sys_metrics.csv")
```

### 비샘플링 메트릭 데이터 가져오기

기본적으로 history에서 데이터를 가져오면 500개 지점 샘플링 됩니다. 기록된 모든 데이터를 가져오려면 `run.scan_history()`를 사용하세요. 여기서는 history에 기록된 모든 `loss` 데이터를 다운로드하는 방법의 예시입니다.

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
history = run.scan_history()
losses = [row["loss"] for row in history]
```

### history에서 페이지네이션 데이터 가져오기

메트릭이 백엔드에서 천천히 가져와지거나 API 요청이 시간 초과하는 경우, `scan_history`에서 페이지 크기를 낮춰 개별 요청이 시간 초과되지 않도록 시도할 수 있습니다. 기본 페이지 크기는 500이며, 어떤 크기가 가장 잘 작동하는지 실험할 수 있습니다:

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.scan_history(keys=sorted(cols), page_size=100)
```

### 프로젝트의 모든 run에서 메트릭을 CSV 파일로 내보내기

이 스크립트는 프로젝트에서 run을 가져와 이름, 설정 및 요약 통계가 포함된 data frame과 run의 CSV를 생성합니다. `<entity>`와 `<project>`를 사용자의 W&B 엔터티와 프로젝트 이름으로 각각 바꾸세요.

```python
import pandas as pd
import wandb

api = wandb.Api()
entity, project = "<entity>", "<project>"
runs = api.runs(entity + "/" + project)

summary_list, config_list, name_list = [], [], []
for run in runs:
    # .summary는 정확도와 같은 메트릭의 출력 키/값을 포함합니다.
    #  ._json_dict를 호출하여 큰 파일을 생략합니다.
    summary_list.append(run.summary._json_dict)

    # .config는 하이퍼파라미터를 포함합니다.
    #  _로 시작하는 특수 값을 제거합니다.
    config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

    # .name은 run의 사람이 읽을 수 있는 이름입니다.
    name_list.append(run.name)

runs_df = pd.DataFrame(
    {"summary": summary_list, "config": config_list, "name": name_list}
)

runs_df.to_csv("project.csv")
```

### run의 시작 시간 가져오기

이 코드조각은 run이 생성된 시간을 검색합니다.

```python
import wandb

api = wandb.Api()

run = api.run("entity/project/run_id")
start_time = run.created_at
```

### 완료된 run에 파일 업로드

아래 코드조각은 완료된 run에 선택한 파일을 업로드합니다.

```python
import wandb

api = wandb.Api()

run = api.run("entity/project/run_id")
run.upload_file("file_name.extension")
```

### run에서 파일 다운로드

이 코드는 run ID `uxte44z7`와 관련된 "model-best.h5" 파일을 찾아 로컬에 저장합니다.

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.file("model-best.h5").download()
```

### run의 모든 파일 다운로드

이 코드는 run에 관련된 모든 파일을 찾아 로컬에 저장합니다.

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
for file in run.files():
    file.download()
```

### 특정 스윕에서 Runs 가져오기

이 코드조각은 특정 스윕과 관련된 모든 run을 다운로드합니다.

```python
import wandb

api = wandb.Api()

sweep = api.sweep("<entity>/<project>/<sweep_id>")
sweep_runs = sweep.runs
```

### 스윕에서 가장 좋은 run 가져오기

다음 코드조각은 주어진 스윕에서 가장 좋은 run을 가져옵니다.

```python
import wandb

api = wandb.Api()

sweep = api.sweep("<entity>/<project>/<sweep_id>")
best_run = sweep.best_run()
```

`best_run`은 스윕 설정에서 `metric` 파라미터로 정의된 최고의 메트릭을 가진 run입니다.

### 스윕에서 가장 좋은 모델 파일 다운로드

이 코드조각은 모델 파일을 `model.h5`로 저장한 runs에서 검증 정확도가 가장 높은 모델 파일을 다운로드합니다.

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

### run에서 특정 확장자를 가진 모든 파일 삭제

이 코드조각은 run에서 주어진 확장자를 가진 파일을 삭제합니다.

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

이 코드조각은 run의 모든 시스템 리소스 소비 메트릭이 포함된 데이터프레임을 생성한 후 CSV에 저장합니다.

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
system_metrics = run.history(stream="events")
system_metrics.to_csv("sys_metrics.csv")
```

### 요약 메트릭 업데이트

사전을 사용하여 요약 메트릭을 업데이트할 수 있습니다.

```python
summary.update({"key": val})
```

### run을 실행한 명령 가져오기

각 run은 run 개요 페이지에서 실행을 시작한 명령을 캡처합니다. API에서 이 명령을 가져오려면 다음을 실행하세요:

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")

meta = json.load(run.file("wandb-metadata.json").download())
program = ["python"] + [meta["program"]] + meta["args"]
```
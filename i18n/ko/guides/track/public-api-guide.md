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

W&B 공개 API를 사용하여 MLFlow 또는 W&B 인스턴스 간에 데이터를 내보내거나 가져옵니다.

## MLFlow에서 데이터 가져오기

W&B는 실험, run, 아티팩트, 메트릭 및 기타 메타데이터를 포함하여 MLFlow에서 데이터를 가져오는 것을 지원합니다.

의존성 설치:

```shell
pip install wandb[importers]
```

W&B에 로그인합니다. 이전에 로그인한 적이 없다면 프롬프트를 따르세요.

```shell
wandb login
```

기존 MLFlow 서버에서 모든 run 가져오기:

```py
from wandb.apis.importers.mlflow import MlflowImporter

importer = MlflowImporter(mlflow_tracking_uri="...")

runs = importer.collect_runs()
importer.import_runs(runs)
```

기본적으로 `importer.collect_runs()`는 MLFlow 서버의 모든 run을 수집합니다. 특별한 서브셋을 업로드하려면 자신만의 runs 이터러블을 구성하여 가져오기에 전달할 수 있습니다.

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
Databricks MLFlow에서 가져올 경우 먼저 [Databricks CLI를 구성](https://docs.databricks.com/dev-tools/cli/index.html)해야 할 수 있습니다.

이전 단계에서 `mlflow-tracking-uri="databricks"`를 설정하세요.
:::

아티팩트를 가져오지 않으려면 `artifacts=False`를 전달할 수 있습니다:

```py
importer.import_runs(runs, artifacts=False)
```

특정 W&B 엔티티 및 프로젝트로 가져오려면 `Namespace`를 전달할 수 있습니다:

```py
from wandb.apis.importers import Namespace

importer.import_runs(runs, namespace=Namespace(entity, project))
```

## 다른 W&B 인스턴스에서 데이터 가져오기

:::info
이 기능은 베타이며, W&B 공용 클라우드에서 가져오기만 지원합니다.
:::

의존성 설치:

```sh
pip install wandb[importers]
```

소스 W&B 서버에 로그인합니다. 이전에 로그인한 적이 없다면 프롬프트를 따르세요.

```sh
wandb login
```

소스 W&B 인스턴스에서 목적지 W&B 인스턴스로 모든 run 및 아티팩트를 가져옵니다. Run 및 아티팩트는 목적지 인스턴스의 해당 네임스페이스로 가져옵니다.

```py
from wandb.apis.importers.wandb import WandbImporter
from wandb.apis.importers import Namespace

importer = WandbImporter(
    src_base_url="https://api.wandb.ai",
    src_api_key="your-api-key-here",
    dst_base_url="https://example-target.wandb.io",
    dst_api_key="target-environment-api-key-here",
)

# "entity/project"의 src에서 dst의 "entity/project"로
# 모든 run, 아티팩트, 리포트를 가져옵니다.
importer.import_all(namespaces=[
    Namespace(entity, project),
    # ... 여기에 더 많은 네임스페이스 추가
])
```

목적지 네임스페이스를 변경하려면 `remapping: dict[Namespace, Namespace]`를 지정할 수 있습니다.

```py
importer.import_all(
    namespaces=[Namespace(entity, project)],
    remapping={
        Namespace(entity, project): Namespace(new_entity, new_project),
    }
)
```

기본적으로 가져오기는 증분적입니다. 후속 가져오기는 이전 작업을 검증하고 성공/실패를 추적하는 `.jsonl` 파일에 쓰려고 시도합니다. 가져오기가 성공하면, 향후 검증이 생략됩니다. 가져오기가 실패하면, 재시도됩니다. 이를 비활성화하려면 `incremental=False`로 설정하세요.

```py
importer.import_all(
    namespaces=[Namespace(entity, project)],
    incremental=False,
)
```

### 알려진 문제 및 제한 사항

- 목적지 네임스페이스가 존재하지 않는 경우, W&B는 자동으로 하나를 생성합니다.
- run 또는 아티팩트가 목적지 네임스페이스에서 동일한 ID를 갖는 경우, W&B는 이를 증분 가져오기로 처리합니다. 목적지 run/아티팩트는 검증되며, 이전 가져오기에서 실패한 경우 재시도됩니다.
- 소스 시스템에서 데이터가 삭제되는 일은 절대 없습니다.

1. 대량 가져오기(특히 큰 아티팩트)를 할 때 때때로 S3 속도 제한에 걸릴 수 있습니다. `botocore.exceptions.ClientError: An error occurred (SlowDown) when calling the PutObject operation`이 표시되면, 한 번에 몇 개의 네임스페이스만 이동하여 가져오기를 간격을 두고 시도할 수 있습니다.
2. 가져온 run 테이블이 워크스페이스에서 비어 보이지만, 아티팩트 탭으로 이동하여 해당 run 테이블 아티팩트를 클릭하면 예상대로 테이블을 볼 수 있습니다.
3. 시스템 메트릭 및 사용자 정의 차트(`wandb.log`으로 명시적으로 로그되지 않은 경우)는 가져오지 않습니다.

## 데이터 내보내기

스크립트에서 데이터를 로그한 후 공개 API를 사용하여 W&B에 저장한 데이터를 내보내거나 업데이트할 수 있습니다. 자세한 내용은 [퀵스타트](../../quickstart.md)를 확인하세요.

**공개 API의 사용 사례**

- **데이터 내보내기**: Jupyter Notebook에서 사용자 지정 분석을 위해 데이터프레임을 가져옵니다. 데이터를 탐색한 후 새로운 분석 run을 생성하고 결과를 로깅하여 발견한 내용을 동기화할 수 있습니다. 예를 들어: `wandb.init(job_type="analysis")`
- **기존 Run 업데이트**: W&B run과 연관된 데이터를 업데이트할 수 있습니다. 예를 들어, 원래 로그되지 않았던 아키텍처 또는 하이퍼파라미터와 같은 추가 정보를 포함하도록 일련의 run의 설정을 업데이트하고 싶을 수 있습니다.

사용 가능한 함수에 대한 자세한 내용은 [생성된 참조 문서](../../ref/python/public-api/README.md)를 참조하세요.

### 인증

다음 두 가지 방법 중 하나로 [API 키](https://wandb.ai/authorize)를 사용하여 컴퓨터를 인증하세요.

1. 커맨드라인에서 `wandb login`을 실행하고 API 키를 붙여넣습니다.
2. `WANDB_API_KEY` 환경 변수를 API 키로 설정합니다.

### run 경로 찾기

Public API를 사용하려면 종종 `<entity>/<project>/<run_id>` 형식의 run 경로가 필요합니다. 앱 UI에서 run 페이지를 열고 [Overview 탭](../app/pages/run-page.md#overview-tab)을 클릭하여 run 경로를 얻습니다.

### Run 데이터 내보내기

완료되었거나 활성 상태인 run에서 데이터를 다운로드합니다. 일반적인 사용 사례에는 Jupyter 노트북에서 사용자 지정 분석을 위한 데이터프레임을 다운로드하거나 자동화된 환경에서 사용자 지정 로직을 사용하는 것이 포함됩니다.

```python
import wandb

api = wandb.Api()
run = api.run("<entity>/<project>/<run_id>")
```

run 객체의 가장 일반적으로 사용되는 속성은 다음과 같습니다:

| 속성             | 의미                                                                                                                                                                                                                                                                                                                 |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `run.config`    | 데이터셋 아티팩트를 생성하는 run의 전처리 방법이나 트레이닝 run의 하이퍼파라미터와 같은 run의 설정 정보를 담은 사전입니다. 이를 run의 "입력"으로 생각하세요.                                                                                                                                      |
| `run.history()` | 모델이 트레이닝되는 동안 변경되는 값들, 예를 들어 손실 같은 값을 저장하기 위한 사전들의 리스트입니다. `wandb.log()` 명령어는 이 객체에 추가합니다.                                                                                                                                                              |
| `run.summary`   | run 결과를 요약하는 정보를 담은 사전입니다. 이는 정확도와 손실과 같은 스칼라 또는 큰 파일일 수 있습니다. 기본적으로, `wandb.log()`는 로깅된 시계열의 최종 값을 요약에 설정합니다. 요약의 내용은 직접 설정할 수도 있습니다. 이를 run의 "출력"으로 생각하세요. |

과거 run의 데이터를 수정하거나 업데이트할 수도 있습니다. 기본적으로 api 객체의 단일 인스턴스는 모든 네트워크 요청을 캐시합니다. 실행 스크립트에서 실시간 정보가 필요한 사용 사례의 경우, 업데이트된 값을 얻기 위해 `api.flush()`를 호출하세요.

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

이는 위 run 객체 속성에 대한 다양한 출력입니다

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

기본적으로 history 메소드는 메트릭을 고정된 수의 샘플(기본값은 500, `samples` 인수로 변경 가능)로 샘플링합니다. 큰 run의 모든 데이터를 내보내려면 `run.scan_history()` 메소드를 사용할 수 있습니다. 자세한 내용은 [API 참조](https://docs.wandb.ai/ref/python/public-api)를 참조하세요.

### 여러 Run 쿼리하기

<Tabs
defaultValue="dataframes_csvs"
values={[
{label: '데이터프레임 및 CSV', value: 'dataframes_csvs'},
{label: 'MongoDB 스타일', value: 'mongoDB'},
]}>
<TabItem value="dataframes_csvs">

이 예제 스크립트는 프로젝트를 찾고 이름, 설정 및 요약 통계가 포함된 runs의 CSV를 출력합니다. `<entity>` 및 `<project>`를 귀하의 W&B 엔티티와 프로젝트 이름으로 각각 교체하세요.

```python
import pandas as pd
import wandb

api = wandb.Api()
entity, project = "<entity>", "<project>"
runs = api.runs(entity + "/" + project)

summary_list, config_list, name_list = [], [], []
for run in runs:
    # .summary에는 정확도와 같은
    # 메트릭의 출력 키/값이 포함됩니다.
    # 대용량 파일을 생략하기 위해 ._json_dict를 호출합니다
    summary_list.append(run.summary._json_dict)

    # .config에는 하이퍼파라미터가 포함됩니다.
    # _로 시작하는 특수 값은 제거합니다.
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

W&B API는 api.runs()를 사용하여 프로젝트의 여러 run을 쿼리할 수 있는 방법을 제공합니다. 가장 일반적인 사용 사례는 사용자 지정 분석을 위한 runs 데이터를 내보내는 것입니다. 쿼리 인터페이스는 [MongoDB가 사용하는 것과](https://docs.mongodb.com/manual/reference/operator/query) 동일합니다.

```python
runs = api.runs(
    "username/project",
    {"$or": [{"config.experiment_name": "foo"}, {"config.experiment_name": "bar"}]},
)
print(f"Found {len(runs)} runs")
```

  </TabItem>
</Tabs>

`api.runs`를 호출하면 리스트처럼 작동하는 반복 가능한 `Runs` 객체가 반환됩니다. 기본적으로 객체는 시퀀스로 필요한 경우 50개의 run을 한 번에 로드하지만, `per_page` 키워드 인수로 로드되는 페이지당 수를 변경할 수 있습니다.

`api.runs`는 또한 `order` 키워드 인수를 수락합니다. 기본 순서는 `-created_at`이며, 오름차순 결과를 얻으려면 `+created_at`을 지정하세요. 또한 config 또는 summary 값으로 정렬할 수 있습니다, 예를 들어 `summary.val_acc` 또는 `config.experiment_name`

### 오류 처리

W&B 서버와 통신하는 동안 오류가 발생하면 `wandb.CommError`가 발생합니다. 원래 예외는 `exc` 속성을 통해 조사할 수 있습니다.

### API를 통해 최신 git 커밋 가져오기

UI에서 run을 클릭한 다음 run 페이지의 Overview 탭을 클릭하여 최신 git 커밋을 확인합니다. 또한 `wandb-metadata.json` 파일에 있습니다. 공개 API를 사용하면 `run.commit`으로 git 해시를 가져올 수 있습니다.

## 자주 묻는 질문들

### matplotlib 또는 seaborn에서 시각화하기 위해 데이터를 내보내려면 어떻게 해야 하나요?

일반적인 내보내기 패턴에 대해서는 [API 예제](../../ref/python/public-api/README.md)를 확인하세요. 사용자 지정 차트나 확장된 runs 테이블에서 다운로드 버튼을 클

### 실행된 코맨드 가져오기

각 실행은 실행 개요 페이지에서 실행을 시작한 코맨드를 캡처합니다. 이 코맨드를 API에서 가져오려면 다음을 실행할 수 있습니다:

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<프로젝트>/<run_id>")

meta = json.load(run.file("wandb-metadata.json").download())
program = ["python"] + [meta["program"]] + meta["args"]
```
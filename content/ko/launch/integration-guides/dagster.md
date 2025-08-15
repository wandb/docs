---
title: Dagster
description: Dagster와 W&B를 연동하는 가이드.
menu:
  launch:
    identifier: ko-launch-integration-guides-dagster
    parent: launch-integration-guides
url: guides/integrations/dagster
---

Dagster와 W&B (Weights & Biases, W&B)를 함께 사용하여 MLOps 파이프라인을 오케스트레이션하고 ML 자산을 효과적으로 관리할 수 있습니다. W&B와의 통합 덕분에 Dagster 내에서 손쉽게 다음과 같은 작업을 할 수 있습니다.

* [W&B Artifact]({{< relref path="/guides/core/artifacts/" lang="ko" >}}) 생성 및 활용
* [W&B Registry]({{< relref path="/guides/core/registry/" lang="ko" >}})에서 Registered Models 사용 및 생성
* [W&B Launch]({{< relref path="/launch/" lang="ko" >}})를 활용한 전용 컴퓨트에서 트레이닝 잡 실행
* ops와 assets에서 [wandb]({{< relref path="/ref/python/" lang="ko" >}}) 클라이언트 사용

W&B Dagster 통합은 W&B 특화 Dagster resource와 IO Manager를 제공합니다:

* `wandb_resource`: W&B API 인증 및 통신에 사용되는 Dagster resource
* `wandb_artifacts_io_manager`: W&B Artifacts를 소비하는 데 사용되는 Dagster IO Manager

아래 가이드에서는 Dagster에서 W&B를 사용하기 위한 사전 준비사항부터, ops 및 assets에서 W&B Artifacts 생성·활용, W&B Launch 활용과 모범 사례까지 단계별로 설명합니다.

## 시작 전 준비 사항
Dagster에서 W&B를 사용하려면 다음 리소스가 필요합니다:
1. **W&B API Key**
2. **W&B entity (user 또는 team)**: entity는 W&B Runs와 Artifacts를 전송할 사용자명 또는 팀명입니다. run을 기록하기 전에 꼭 W&B App UI에서 계정 또는 팀 entity를 생성하세요. entity를 지정하지 않으면 run은 기본 entity(보통 본인 username)에 기록됩니다. 기본 entity는 **Project Defaults** 아래의 settings에서 변경할 수 있습니다.
3. **W&B project**: [W&B Runs]({{< relref path="/guides/models/track/runs/" lang="ko" >}})가 저장될 프로젝트명

본인의 W&B entity는 W&B App에서 해당 사용자나 팀의 프로필 페이지에서 확인할 수 있습니다. 기존 프로젝트를 쓰거나 새 프로젝트를 만들 수 있으며, App의 홈 또는 프로필 페이지에서 새 프로젝트를 생성할 수 있습니다. 프로젝트가 없다면 처음 사용 시 자동으로 생성됩니다. 아래는 API 키를 얻는 방법을 안내합니다:

### API 키 획득 방법
1. [W&B에 로그인](https://wandb.ai/login)하세요. 참고: W&B Server를 사용하는 경우 인스턴스 호스트명을 관리자로부터 문의하세요.
2. [authorize 페이지](https://wandb.ai/authorize) 또는 사용자/팀 settings에서 API 키를 확인하세요. 프로덕션 환경이라면 해당 키는 [service account]({{< relref path="/support/kb-articles/service_account_useful.md" lang="ko" >}})를 사용하는 것을 권장합니다.
3. 환경변수로 API 키를 등록하세요. 예시: `WANDB_API_KEY=YOUR_KEY`

아래 예제에서는 Dagster 코드에서 API 키 지정 위치를 안내합니다. 반드시 `wandb_config` 딕셔너리 내에 entity와 project 이름을 지정하세요. 여러 ops/assets에서 서로 다른 W&B Project를 사용하고 싶으면 각기 다른 `wandb_config` 값을 전달할 수 있습니다. 전달 가능한 키에 대해 더 알고 싶으면 아래 Configuration 섹션을 참고하세요.


{{< tabpane text=true >}}
{{% tab "Config for @job" %}}
예시: `@job`에 대한 설정 예시
```python
# config.yaml에 아래 내용을 추가하세요.
# 또는 Dagit의 Launchpad, JobDefinition.execute_in_process에서 설정할 수 있습니다.
# 참고: https://docs.dagster.io/concepts/configuration/config-schema#specifying-runtime-configuration
resources:
 wandb_config:
   config:
     entity: my_entity # 본인의 W&B entity로 수정
     project: my_project # 본인의 W&B project로 수정


@job(
   resource_defs={
       "wandb_config": make_values_resource(
           entity=str,
           project=str,
       ),
       "wandb_resource": wandb_resource.configured(
           {"api_key": {"env": "WANDB_API_KEY"}}
       ),
       "io_manager": wandb_artifacts_io_manager,
   }
)
def simple_job_example():
   my_op()
```
{{% /tab %}}
{{% tab "Config for @repository using assets" %}}

예시: assets와 함께 사용하는 `@repository` 설정 예시

```python
from dagster_wandb import wandb_artifacts_io_manager, wandb_resource
from dagster import (
   load_assets_from_package_module,
   make_values_resource,
   repository,
   with_resources,
)

from . import assets

@repository
def my_repository():
   return [
       *with_resources(
           load_assets_from_package_module(assets),
           resource_defs={
               "wandb_config": make_values_resource(
                   entity=str,
                   project=str,
               ),
               "wandb_resource": wandb_resource.configured(
                   {"api_key": {"env": "WANDB_API_KEY"}}
               ),
               "wandb_artifacts_manager": wandb_artifacts_io_manager.configured(
                   {"cache_duration_in_minutes": 60} # 1시간만 캐시 유지
               ),
           },
           resource_config_by_key={
               "wandb_config": {
                   "config": {
                       "entity": "my_entity", # 본인의 W&B entity로 수정
                       "project": "my_project", # 본인의 W&B project로 수정
                   }
               }
           },
       ),
   ]
```
이 예시에선 IO Manager의 캐시 유지 시간을 설정하는 점이 `@job` 예제와 다릅니다.
{{% /tab %}}
{{< /tabpane >}}

### Configuration
아래 설정 옵션들은 W&B 전용 Dagster resource와 IO Manager를 구성할 때 사용합니다.

* `wandb_resource`: [resource](https://docs.dagster.io/concepts/resources)로서, W&B API와 통신하며, 제공된 API 키로 자동 인증됩니다.
    * `api_key`: (str, 필수) W&B API와 통신에 필요한 키입니다.
    * `host`: (str, 선택) 사용할 API 호스트 서버. W&B Server를 사용할 때만 필요합니다. 기본값은 Public Cloud의 `https://api.wandb.ai`입니다.
* `wandb_artifacts_io_manager`: [IO Manager](https://docs.dagster.io/concepts/io-management/io-managers)로서, W&B Artifacts를 사용합니다.
    * `base_dir`: (int, 선택) 로컬 저장·캐시에 사용할 기준 디렉토리. 기본값은 `DAGSTER_HOME`입니다.
    * `cache_duration_in_minutes`: (int, 선택) W&B Artifacts와 Run 로그를 로컬에 유지할 시간(분). 이 시간이 지난 후 열지 않은 파일/폴더는 캐시에서 삭제됩니다. 캐시 비활성화는 0으로 설정하세요. 기본값은 30일입니다.
    * `run_id`: (str, 선택) 이 run을 구분하는 고유 ID. 프로젝트 내에서 유일해야 하며, run 삭제 시 동일 ID를 재사용할 수 없습니다. 이름 필드에는 run의 간단 설명을, config에는 하이퍼파라미터 저장 등을 활용할 수 있습니다. ID에는 `/\#?%:..` 특수문자가 불가합니다. Dagster 내 experiment tracking 진행 시 IO Manager가 run 재개(resume)를 하려면 꼭 Run ID를 설정하세요. 기본값으로는 Dagster Run ID가 사용됩니다(ex: `7e4df022-1bf2-44b5-a383-bb852df4077e`).
    * `run_name`: (str, 선택) UI에서 run을 구분하는 간단한 표시 이름. 기본값은 `dagster-run-[Dagster Run ID 앞 8자리]`입니다. 예시: `dagster-run-7e4df022`
    * `run_tags`: (list[str], 선택) run에 추가될 태그 리스트. 태그는 run 정리에 유용하며, `baseline` 또는 `production`과 같은 임시 라벨에도 활용할 수 있습니다. UI에서 태그 추가/삭제 및 필터링이 가능합니다. 통합에서 사용하는 모든 Run에는 `dagster_wandb` 태그가 부여됩니다.

## W&B Artifacts 사용하기

W&B Artifact 연동은 Dagster IO Manager로 작동합니다.

[IO Manager](https://docs.dagster.io/concepts/io-management/io-managers)는 asset 또는 op의 결과를 저장·로딩하는 역할을 가진 사용자 제공 오브젝트입니다. 예를 들어 IO Manager는 파일 시스템에서 오브젝트를 저장·로드할 수 있습니다.

본 통합은 W&B Artifacts를 위한 IO Manager를 제공합니다. 이로써 어떤 Dagster `@op` 또는 `@asset`이든 네이티브하게 W&B Artifact 생성과 소비가 가능합니다. 아래는 Python 리스트를 dataset 타입의 W&B Artifact로 생성하는 간단한 `@asset` 예시입니다.

```python
@asset(
    name="my_artifact",
    metadata={
        "wandb_artifact_arguments": {
            "type": "dataset",
        }
    },
    io_manager_key="wandb_artifacts_manager",
)
def create_dataset():
    return [1, 2, 3] # 이 값이 Artifact로 저장됩니다
```

`@op`, `@asset`, `@multi_asset`마다 metadata에 Artifacts 기록 설정을 지정할 수 있습니다. 반대로 W&B 이외에서 생성된 Artifacts도 소비할 수 있습니다.

## W&B Artifacts 기록하기
진행 전에, W&B Artifacts 사용법을 충분히 이해하고 있는 것이 좋습니다. [Artifacts 가이드]({{< relref path="/guides/core/artifacts/" lang="ko" >}})를 참고하세요.

Python 함수에서 오브젝트를 반환하면 W&B Artifact로 저장됩니다. 지원 오브젝트는 다음과 같습니다:
* Python 오브젝트(int, dict, list 등)
* W&B 오브젝트(Table, Image, Graph 등)
* W&B Artifact 오브젝트

아래 예시는 Dagster asset(`@asset`)으로 W&B Artifact를 작성하는 방법을 보여줍니다:


{{< tabpane text=true >}}
{{% tab "Python objects" %}}
[pickle](https://docs.python.org/3/library/pickle.html) 모듈로 직렬화할 수 있는 모든 오브젝트는 integration에서 Artifact에 추가되며, Dagster 내에서 Artifact를 읽을 때 자동으로 unpickle 됩니다([Artifact 읽기]({{< relref path="#read-wb-artifacts" lang="ko" >}}) 참고).

```python
@asset(
    name="my_artifact",
    metadata={
        "wandb_artifact_arguments": {
            "type": "dataset",
        }
    },
    io_manager_key="wandb_artifacts_manager",
)
def create_dataset():
    return [1, 2, 3]
```

W&B는 여러 pickle 기반 직렬화 모듈([pickle](https://docs.python.org/3/library/pickle.html), [dill](https://github.com/uqfoundation/dill), [cloudpickle](https://github.com/cloudpipe/cloudpickle), [joblib](https://github.com/joblib/joblib))을 지원합니다. [ONNX](https://onnx.ai/)나 [PMML](https://en.wikipedia.org/wiki/Predictive_Model_Markup_Language) 같은 고급 직렬화도 가능합니다. 자세한 내용은 [직렬화 설정]({{< relref path="#serialization-configuration" lang="ko" >}}) 항목을 참고하세요.
{{% /tab %}}
{{% tab "W&B Object" %}}
[Table]({{< relref path="/ref/python/sdk/data-types/table.md" lang="ko" >}})이나 [Image]({{< relref path="/ref/python/sdk/data-types/image.md" lang="ko" >}})처럼 W&B 오브젝트도 integration을 통해 Artifact에 추가할 수 있습니다. 아래 예시에서는 Table 오브젝트를 Artifact로 추가합니다:

```python
import wandb

@asset(
    name="my_artifact",
    metadata={
        "wandb_artifact_arguments": {
            "type": "dataset",
        }
    },
    io_manager_key="wandb_artifacts_manager",
)
def create_dataset_in_table():
    return wandb.Table(columns=["a", "b", "c"], data=[[1, 2, 3]])
```

{{% /tab %}}
{{% tab "W&B Artifact" %}}

복잡한 유스 케이스에서는 직접 Artifact 오브젝트를 만들어야 할 수도 있습니다. 그래도 integration에서는 양쪽에 메타데이터를 보강해주는 등 유용한 부가기능을 유지합니다.

```python
import wandb

MY_ASSET = "my_asset"

@asset(
    name=MY_ASSET,
    io_manager_key="wandb_artifacts_manager",
)
def create_artifact():
   artifact = wandb.Artifact(MY_ASSET, "dataset")
   table = wandb.Table(columns=["a", "b", "c"], data=[[1, 2, 3]])
   artifact.add(table, "my_table")
   return artifact
```
{{% /tab %}}
{{< /tabpane >}}

### Configuration
`wandb_artifact_configuration`라는 설정 딕셔너리를 `@op`, `@asset`, `@multi_asset`에서 메타데이터(데코레이터 인자)에 지정할 수 있습니다. 이 설정은 W&B Artifacts의 IO Manager 읽기/쓰기 제어에 반드시 필요합니다.

`@op`의 경우 [Out](https://docs.dagster.io/_apidocs/ops#dagster.Out) metadata 인자, 
`@asset`의 경우 asset의 metadata 인자, 
`@multi_asset`의 경우 [AssetOut](https://docs.dagster.io/_apidocs/assets#dagster.AssetOut) metadata 인자 내에 각각 위치합니다.

아래 코드는 `@op`, `@asset`, `@multi_asset`의 설정 방법을 보여줍니다:

{{< tabpane text=true >}}
{{% tab "Example for @op" %}}
`@op` 예시:
```python 
@op(
   out=Out(
       metadata={
           "wandb_artifact_configuration": {
               "name": "my_artifact",
               "type": "dataset",
           }
       }
   )
)
def create_dataset():
   return [1, 2, 3]
```
{{% /tab %}}
{{% tab "Example for @asset" %}}
`@asset` 예시:
```python
@asset(
   name="my_artifact",
   metadata={
       "wandb_artifact_configuration": {
           "type": "dataset",
       }
   },
   io_manager_key="wandb_artifacts_manager",
)
def create_dataset():
   return [1, 2, 3]
```

이 경우 설정에서 name을 별도로 지정하지 않아도 됩니다. integration이 asset명을 Artifact명으로 자동 설정합니다.

{{% /tab %}}
{{% tab "Example for @multi_asset" %}}

`@multi_asset` 예시:

```python
@multi_asset(
   name="create_datasets",
   outs={
       "first_table": AssetOut(
           metadata={
               "wandb_artifact_configuration": {
                   "type": "training_dataset",
               }
           },
           io_manager_key="wandb_artifacts_manager",
       ),
       "second_table": AssetOut(
           metadata={
               "wandb_artifact_configuration": {
                   "type": "validation_dataset",
               }
           },
           io_manager_key="wandb_artifacts_manager",
       ),
   },
   group_name="my_multi_asset_group",
)
def create_datasets():
   first_table = wandb.Table(columns=["a", "b", "c"], data=[[1, 2, 3]])
   second_table = wandb.Table(columns=["d", "e"], data=[[4, 5]])

   return first_table, second_table
```
{{% /tab %}}
{{< /tabpane >}}

지원 속성들:
* `name`: (str) 이 Artifact를 UI에서 식별하거나 use_artifact 호출 시 참조하는 사람 친화적 이름입니다. 프로젝트 내에서 유일해야 하며, `@op`에서 필수입니다.
* `type`: (str) Artifact 타입. dataset, model 등이 대표적이지만, 영문/숫자/밑줄/하이픈/점으로 자유롭게 지정 가능합니다. 아웃풋이 이미 Artifact가 아니라면 필수.
* `description`: (str) Artifact에 대한 자유로운 설명. UI에서는 markdown 랜더링되므로 표나 링크 등 추가하기에 유용합니다.
* `aliases`: (list[str]) Artifact에 적용할 별칭들의 배열입니다. “latest”는 기본적으로 추가됩니다. 모델과 데이터셋 버전 관리를 편리하게 할 수 있습니다.
* [`add_dirs`]({{< relref path="/ref/python/sdk/classes/artifact#add_dir" lang="ko" >}}): (list[dict[str, Any]]) Artifact에 추가할 로컬 디렉토리들의 배열
* [`add_files`]({{< relref path="/ref/python/sdk/classes/artifact#add_file" lang="ko" >}}): (list[dict[str, Any]]) Artifact에 추가할 로컬 파일 설정 배열
* [`add_references`]({{< relref path="/ref/python/sdk/classes/artifact#add_reference" lang="ko" >}}): (list[dict[str, Any]]) Artifact에 추가할 외부 참조 배열 (예: URL, S3 등)
* `serialization_module`: (dict) 사용할 직렬화 모듈 세부 설정
    * `name`: (str) 직렬화 모듈명(`pickle`, `dill`, `cloudpickle`, `joblib`). 모듈이 로컬에 설치되어 있어야 합니다.
    * `parameters`: (dict[str, Any]) 선택적 파라미터, 각 모듈의 dump 메소드 파라미터와 동일. 예: `{"compress": 3, "protocol": 4}`.

고급 예시:

```python
@asset(
   name="my_advanced_artifact",
   metadata={
       "wandb_artifact_configuration": {
           "type": "dataset",
           "description": "My *Markdown* description",
           "aliases": ["my_first_alias", "my_second_alias"],
           "add_dirs": [
               {
                   "name": "My directory",
                   "local_path": "path/to/directory",
               }
           ],
           "add_files": [
               {
                   "name": "validation_dataset",
                   "local_path": "path/to/data.json",
               },
               {
                   "is_tmp": True,
                   "local_path": "path/to/temp",
               },
           ],
           "add_references": [
               {
                   "uri": "https://picsum.photos/200/300",
                   "name": "External HTTP reference to an image",
               },
               {
                   "uri": "s3://my-bucket/datasets/mnist",
                   "name": "External S3 reference",
               },
           ],
       }
   },
   io_manager_key="wandb_artifacts_manager",
)
def create_advanced_artifact():
   return [1, 2, 3]
```

Asset이 integration 양쪽에서 유용한 메타데이터와 함께 materialize 됩니다:
* W&B 측: 소스 integration 이름과 버전, 사용한 python 버전, pickle 프로토콜 버전 등
* Dagster 측:
    * Dagster Run ID
    * W&B Run의 ID, 이름, 경로, URL
    * W&B Artifact의 ID, 이름, 타입, 버전, 크기, URL
    * W&B Entity, W&B Project

아래 이미지는 integration을 통해 Dagster asset에 추가된 W&B의 메타데이터를 보여줍니다. 이 정보는 integration이 없다면 확인할 수 없습니다.

{{< img src="/images/integrations/dagster_wb_metadata.png" alt="" >}}

아래 이미지는 제공된 설정이 W&B Artifact 메타데이터와 함께 잘 보강되어 저장되는 예를 보여줍니다. 이는 재현성, 유지보수에 큰 도움이 됩니다.

{{< img src="/images/integrations/dagster_inte_1.png" alt="" >}}
{{< img src="/images/integrations/dagster_inte_2.png" alt="" >}}
{{< img src="/images/integrations/dagster_inte_3.png" alt="" >}}

{{% alert %}}
mypy 같은 정적 타입체커를 사용할 경우, 설정 타입 정의 오브젝트를 아래와 같이 import 하세요:

```python
from dagster_wandb import WandbArtifactConfiguration
```
{{% /alert %}}

### 파티션 사용

이 통합은 [Dagster 파티션](https://docs.dagster.io/concepts/partitions-schedules-sensors/partitions)을 기본 지원합니다.

아래는 `DailyPartitionsDefinition`으로 파티셔닝한 예입니다.
```python
@asset(
    partitions_def=DailyPartitionsDefinition(start_date="2023-01-01", end_date="2023-02-01"),
    name="my_daily_partitioned_asset",
    compute_kind="wandb",
    metadata={
        "wandb_artifact_configuration": {
            "type": "dataset",
        }
    },
)
def create_my_daily_partitioned_asset(context):
    partition_key = context.asset_partition_key_for_output()
    context.log.info(f"Creating partitioned asset for {partition_key}")
    return random.randint(0, 100)
```
이 코드는 각 파티션마다 W&B Artifact를 생성합니다. Artifact 패널(UI)에서 asset명 뒤에 파티션 키가 붙은 형태로 아티팩트를 확인할 수 있습니다. 예: `my_daily_partitioned_asset.2023-01-01`, `my_daily_partitioned_asset.2023-01-02`, `my_daily_partitioned_asset.2023-01-03` 등. 다차원 파티션의 경우 점(.)으로 구분됩니다(예: `my_asset.car.blue`).

{{% alert color="secondary" %}}
한 run 안에서 여러 파티션을 materialize 하는 것은 지원하지 않습니다. 여러 run을 실행해 자산을 materialize 해야 하며, Dagit에서 자산 materialize 시 실행할 수 있습니다.

{{< img src="/images/integrations/dagster_multiple_runs.png" alt="" >}}
{{% /alert %}}

#### 고급 사용법
- [Partitioned job 예시](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/ops/partitioned_job.py)
- [Simple partitioned asset 예시](https://github.com/wandb/dagster/blob/master/examples/with_wandb/with_wandb/assets/simple_partitions_example.py)
- [Multi-partitioned asset 예시](https://github.com/wandb/dagster/blob/master/examples/with_wandb/with_wandb/assets/multi_partitions_example.py)
- [고급 파티션 예시](https://github.com/wandb/dagster/blob/master/examples/with_wandb/with_wandb/assets/advanced_partitions_example.py)

## W&B Artifacts 읽기
W&B Artifacts를 읽는 방법도 작성과 유사합니다. `wandb_artifact_configuration`이라는 설정 딕셔너리를 `@op` 또는 `@asset`의 입력에 지정해야 하는 점만 다릅니다.

`@op`에서는 [In](https://docs.dagster.io/_apidocs/ops#dagster.In) metadata 인자에 위치하며, Artifact명을 반드시 명시해야 합니다.

`@asset`에서는 [Asset](https://docs.dagster.io/_apidocs/assets#dagster.AssetIn) In metadata 인자에 위치하며, Artifact명은 지정하지 않아도 됩니다. (상위 asset명과 동일해야 함)

외부에서 생성된 Artifact와 의존성을 맺으려면 [SourceAsset](https://docs.dagster.io/_apidocs/assets#dagster.SourceAsset)을 사용해야 하며, 항상 해당 asset의 latest 버전을 읽습니다.

아래는 다양한 op에서 Artifact를 읽는 예시입니다.

{{< tabpane text=true >}}
{{% tab "From an @op" %}}
`@op`에서 Artifact 읽기
```python
@op(
   ins={
       "artifact": In(
           metadata={
               "wandb_artifact_configuration": {
                   "name": "my_artifact",
               }
           }
       )
   },
   io_manager_key="wandb_artifacts_manager"
)
def read_artifact(context, artifact):
   context.log.info(artifact)
```
{{% /tab %}}
{{% tab "Created by another @asset" %}}
다른 `@asset`이 생성한 Artifact 읽기
```python
@asset(
   name="my_asset",
   ins={
       "artifact": AssetIn(
           # 입력 인자명을 바꾸지 않는다면 'key'를 생략할 수 있습니다
           key="parent_dagster_asset_name",
           input_manager_key="wandb_artifacts_manager",
       )
   },
)
def read_artifact(context, artifact):
   context.log.info(artifact)
```

{{% /tab %}}
{{% tab "Artifact created outside Dagster" %}}

Dagster 외부에서 생성된 Artifact 읽기 예시:

```python
my_artifact = SourceAsset(
   key=AssetKey("my_artifact"),  # W&B Artifact 이름
   description="Artifact created outside Dagster",
   io_manager_key="wandb_artifacts_manager",
)


@asset
def read_artifact(context, my_artifact):
   context.log.info(my_artifact)
```
{{% /tab %}}
{{< /tabpane >}}

### Configuration
아래 설정을 통해 IO Manager가 어떤 데이터를 입력으로 수집해 함수로 제공할지 지정합니다. 주요 읽기 방식은 다음과 같습니다.

1. Artifact 내에 포함된 특정 named object를 얻으려면 get 사용:

```python
@asset(
   ins={
       "table": AssetIn(
           key="my_artifact_with_table",
           metadata={
               "wandb_artifact_configuration": {
                   "get": "my_table",
               }
           },
           input_manager_key="wandb_artifacts_manager",
       )
   }
)
def get_table(context, table):
   context.log.info(table.get_column("a"))
```

2. Artifact 내 다운로드된 파일의 로컬 경로를 얻으려면 get_path 사용:

```python
@asset(
   ins={
       "path": AssetIn(
           key="my_artifact_with_file",
           metadata={
               "wandb_artifact_configuration": {
                   "get_path": "name_of_file",
               }
           },
           input_manager_key="wandb_artifacts_manager",
       )
   }
)
def get_path(context, path):
   context.log.info(path)
```

3. 전체 Artifact 오브젝트(로컬에 다운로드된 상태)를 얻으려면:
```python
@asset(
   ins={
       "artifact": AssetIn(
           key="my_artifact",
           input_manager_key="wandb_artifacts_manager",
       )
   },
)
def get_artifact(context, artifact):
   context.log.info(artifact.name)
```

지원 속성:
* `get`: (str) artifact 내 상대 이름으로 지정된 W&B 오브젝트를 반환
* `get_path`: (str) artifact 내 특정 파일의 로컬 경로 반환

### 직렬화 설정(Serialization configuration)
통합 기본값은 [pickle](https://docs.python.org/3/library/pickle.html) 모듈입니다. 하지만 몇몇 오브젝트(예: yield가 있는 함수)는 pickle로 직렬화할 수 없습니다.

추가로 다른 Pickle 기반 직렬화 모듈([dill](https://github.com/uqfoundation/dill), [cloudpickle](https://github.com/cloudpipe/cloudpickle), [joblib](https://github.com/joblib/joblib))도 쓸 수 있고, [ONNX](https://onnx.ai/) 또는 [PMML](https://en.wikipedia.org/wiki/Predictive_Model_Markup_Language) 등의 고급 포맷도 사용할 수 있습니다(직렬화 문자열을 반환하거나 직접 Artifact 생성). 선택은 유스 케이스에 따라 다르니 관련 자료를 참고하세요.

### Pickle 기반 직렬화 모듈

{{% alert color="secondary" %}}
Pickle 사용은 보안상 안전하지 않을 수 있습니다. 보안이 중요할 경우 W&B 오브젝트만 사용하는 것이 좋으며, 데이터를 서명해서 그 해시 키를 본인 시스템에 저장하길 권장합니다. 복잡한 유스 케이스는 W&B로 문의해주세요.
{{% /alert %}}

`wandb_artifact_configuration` 내 `serialization_module` 딕셔너리로 사용할 직렬화 모듈을 설정할 수 있습니다. Dagster를 실행하는 머신에 모듈이 설치되어 있어야 합니다.

통합에서는 Artifact를 읽을 때 어떤 직렬화 모듈이 사용됐는지 자동 판별할 수 있습니다.

지원되는 모듈은 `pickle`, `dill`, `cloudpickle`, `joblib`입니다.

아래는 `joblib`으로 모델을 직렬화하고 추론에 사용하는 간단 예제입니다.

```python
@asset(
    name="my_joblib_serialized_model",
    compute_kind="Python",
    metadata={
        "wandb_artifact_configuration": {
            "type": "model",
            "serialization_module": {
                "name": "joblib"
            },
        }
    },
    io_manager_key="wandb_artifacts_manager",
)
def create_model_serialized_with_joblib():
    # 실제 ML 모델이 아니지만 pickle 모듈로는 불가능한 예시입니다.
    return lambda x, y: x + y

@asset(
    name="inference_result_from_joblib_serialized_model",
    compute_kind="Python",
    ins={
        "my_joblib_serialized_model": AssetIn(
            input_manager_key="wandb_artifacts_manager",
        )
    },
    metadata={
        "wandb_artifact_configuration": {
            "type": "results",
        }
    },
    io_manager_key="wandb_artifacts_manager",
)
def use_model_serialized_with_joblib(
    context: OpExecutionContext, my_joblib_serialized_model
):
    inference_result = my_joblib_serialized_model(1, 2)
    context.log.info(inference_result)  # 결과: 3 출력
    return inference_result
```

### 고급 직렬화 포맷(ONNX, PMML)
ONNX, PMML과 같은 교환 포맷도 자주 사용됩니다. 통합에서도 지원하지만 Pickle과 비교해 다소 절차가 추가됩니다.

방법은 2가지입니다.
1. 모델을 원하는 포맷으로 변환 후 문자열로 반환하면, integration에서 이를 pickle로 직렬화합니다. 나중에 모델 재구성시 해당 문자열을 사용하면 됩니다.
2. 직렬화된 모델을 파일로 저장한 뒤, add_file 구성으로 커스텀 Artifact를 생성합니다.

Scikit-learn 모델을 ONNX로 직렬화하는 예시입니다.

```python
import numpy
import onnxruntime as rt
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from dagster import AssetIn, AssetOut, asset, multi_asset

@multi_asset(
    compute_kind="Python",
    outs={
        "my_onnx_model": AssetOut(
            metadata={
                "wandb_artifact_configuration": {
                    "type": "model",
                }
            },
            io_manager_key="wandb_artifacts_manager",
        ),
        "my_test_set": AssetOut(
            metadata={
                "wandb_artifact_configuration": {
                    "type": "test_set",
                }
            },
            io_manager_key="wandb_artifacts_manager",
        ),
    },
    group_name="onnx_example",
)
def create_onnx_model():
    # https://onnx.ai/sklearn-onnx/ 참고

    # 모델 학습
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clr = RandomForestClassifier()
    clr.fit(X_train, y_train)

    # ONNX 포맷으로 변환
    initial_type = [("float_input", FloatTensorType([None, 4]))]
    onx = convert_sklearn(clr, initial_types=initial_type)

    # artifacts 기록(model + test_set)
    return onx.SerializeToString(), {"X_test": X_test, "y_test": y_test}

@asset(
    name="experiment_results",
    compute_kind="Python",
    ins={
        "my_onnx_model": AssetIn(
            input_manager_key="wandb_artifacts_manager",
        ),
        "my_test_set": AssetIn(
            input_manager_key="wandb_artifacts_manager",
        ),
    },
    group_name="onnx_example",
)
def use_onnx_model(context, my_onnx_model, my_test_set):
    # https://onnx.ai/sklearn-onnx/ 참고

    # ONNX Runtime으로 예측
    sess = rt.InferenceSession(my_onnx_model)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    pred_onx = sess.run(
        [label_name], {input_name: my_test_set["X_test"].astype(numpy.float32)}
    )[0]
    context.log.info(pred_onx)
    return pred_onx
```

### 파티션 사용

통합은 [Dagster 파티션](https://docs.dagster.io/concepts/partitions-schedules-sensors/partitions)을 기본 지원합니다.

asset의 하나, 여러 개 또는 모든 파티션을 선택적으로 읽을 수 있습니다.

모든 파티션은 사전(dictionary) 형태로 주어지며, key와 value가 각각 파티션 키와 Artifact의 내용을 의미합니다.


{{< tabpane text=true >}}
{{% tab "Read all partitions" %}}
업스트림 `@asset`의 모든 파티션을 dictionary로 읽습니다. 각 key/value는 파티션 키와 Artifact 내용을 의미합니다.
```python
@asset(
    compute_kind="wandb",
    ins={"my_daily_partitioned_asset": AssetIn()},
    output_required=False,
)
def read_all_partitions(context, my_daily_partitioned_asset):
    for partition, content in my_daily_partitioned_asset.items():
        context.log.info(f"partition={partition}, content={content}")
```
{{% /tab %}}
{{% tab "Read specific partitions" %}}
`AssetIn`의 `partition_mapping` 설정으로 특정 파티션만 선택할 수 있습니다. 여기서는 `TimeWindowPartitionMapping`을 사용합니다.
```python
@asset(
    partitions_def=DailyPartitionsDefinition(start_date="2023-01-01", end_date="2023-02-01"),
    compute_kind="wandb",
    ins={
        "my_daily_partitioned_asset": AssetIn(
            partition_mapping=TimeWindowPartitionMapping(start_offset=-1)
        )
    },
    output_required=False,
)
def read_specific_partitions(context, my_daily_partitioned_asset):
    for partition, content in my_daily_partitioned_asset.items():
        context.log.info(f"partition={partition}, content={content}")
```
{{% /tab %}}
{{< /tabpane >}}

설정 오브젝트인 `metadata`는 W&B가 프로젝트 내 다양한 artifact 파티션과 상호작용하는 방법을 지정합니다.

오브젝트 `metadata`에는 `wandb_artifact_configuration` 키가 있으며, 그 하위에 `partitions` 오브젝트가 있습니다.

`partitions` 오브젝트는 각 파티션명마다 설정을 매핑합니다. 각 파티션 설정에는 데이터 수집 방법을 지정하는 `get`, `version`, `alias`와 같은 키가 포함될 수 있습니다.

**설정 키 설명**

1. `get`:  
가져올 데이터와 관련한 W&B 오브젝트명(Table, Image 등) 지정  
2. `version`:  
특정 버전의 Artifact를 참고할 때 사용  
3. `alias`:  
별칭(alias)으로 Artifact를 가져올 때 사용

**와일드카드 설정**

와일드카드 `"*"`는 명시적으로 지정되지 않은 모든 파티션을 의미합니다. non-configured 파티션에 대한 기본 설정을 제공할 수 있습니다.

예시:

```python
"*": {
    "get": "default_table_name",
},
```
이 설정은 명시되지 않은 모든 파티션은 `default_table_name` 테이블에서 데이터를 가져오도록 합니다.

**특정 파티션별 개별 설정**

특정 파티션별로 고유 설정을 추가하여 와일드카드 설정을 덮어쓸 수 있습니다.

예시:

```python
"yellow": {
    "get": "custom_table_name",
},
```

즉, `"yellow"` 파티션은 `custom_table_name` 테이블에서 데이터를 가져옵니다.

**버전 관리 및 별칭**

버전 관리를 위해서는:

```python
"orange": {
    "version": "v0",
},
```

`orange` 파티션의 v0 버전 데이터를 가져오게 됩니다.

별칭을 통한 접근 예시는:

```python
"blue": {
    "alias": "special_alias",
},
```

즉, 별칭 `special_alias`로 참조된 `blue` Artifact 파티션의 `default_table_name` 테이블에서 데이터를 가져옵니다.

### 고급 사용 예시
고급 통합 사용법은 아래 전체 코드 예시를 참고하세요:
* [자세한 예시(assets)](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/assets/advanced_example.py)
* [파티션 잡 예시](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/ops/partitioned_job.py)
* [Model Registry와 모델 연결 예시](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/assets/model_registry_example.py)


## W&B Launch 사용하기

{{% alert color="secondary" %}}
현재 개발 중인 베타 제품입니다.
Launch에 관심이 있다면 계정 담당자에게 연락해서 W&B Launch 고객 파일럿 프로그램 참여를 문의해 주세요.
파일럿 고객은 AWS EKS 또는 SageMaker 환경이 필요합니다. 향후 더 다양한 플랫폼 지원 예정입니다.
{{% /alert %}}

진행 전에, W&B Launch 사용법을 충분히 숙지하는 것이 좋습니다. [Launch 가이드]({{< relref path="/launch/" lang="ko" >}})를 참고하세요.

Dagster 통합에서는 다음을 지원합니다:
* Dagster 인스턴스 내 Launch agent 하나 또는 여러 개 실행
* Dagster 인스턴스 내 로컬 Launch job 실행
* 온프레미스 또는 클라우드에서 원격 Launch job 실행

### Launch agents
통합 모듈은 import 가능한 `@op`인 `run_launch_agent`를 제공합니다. 이 op는 Launch Agent를 시작하며 수동 중단까지 장기 실행 프로세스로 유지됩니다.

Agent는 launch queue를 폴링하고(job 실행 또는 외부 서비스 전송) 잡을 순차적으로 실행하는 프로세스입니다.

자세한 정보는 [Launch 페이지]({{< relref path="/launch/" lang="ko" >}})를 참고하세요.

Launchpad에서도 속성별 유용한 설명을 확인할 수 있습니다.

{{< img src="/images/integrations/dagster_launch_agents.png" alt="" >}}

간단 예시:
```python
# config.yaml에 추가
# Dagit Launchpad, JobDefinition.execute_in_process에서도 설정 가능
# 참고: https://docs.dagster.io/concepts/configuration/config-schema#specifying-runtime-configuration
resources:
 wandb_config:
   config:
     entity: my_entity # 본인의 W&B entity로 수정
     project: my_project # 본인의 W&B project로 수정
ops:
 run_launch_agent:
   config:
     max_jobs: -1
     queues: 
       - my_dagster_queue

from dagster_wandb.launch.ops import run_launch_agent
from dagster_wandb.resources import wandb_resource

from dagster import job, make_values_resource

@job(
   resource_defs={
       "wandb_config": make_values_resource(
           entity=str,
           project=str,
       ),
       "wandb_resource": wandb_resource.configured(
           {"api_key": {"env": "WANDB_API_KEY"}}
       ),
   },
)
def run_launch_agent_example():
   run_launch_agent()
```

### Launch jobs
통합 모듈은 import 가능한 `@op`인 `run_launch_job`도 제공합니다. 해당 op가 Launch job을 실행합니다.

Launch job은 실행을 위해 큐(queue)에 할당됩니다. 직접 큐를 만들거나 기본 큐를 사용할 수 있습니다. 해당 큐를 청취하는 활성 agent가 필요합니다. agent는 Dagster 인스턴스 내에서 돌릴 수도 있고, Kubernetes 배포 agent 활용도 가능합니다.

자세한 정보는 [Launch 페이지]({{< relref path="/launch/" lang="ko" >}})를 참고하세요.

Launchpad에서도 속성별 유용한 설명이 나와 있습니다.

{{< img src="/images/integrations/dagster_launch_jobs.png" alt="" >}}

간단 예시:
```python
# config.yaml에 추가
# Dagit Launchpad, JobDefinition.execute_in_process에서도 설정 가능
# 참고: https://docs.dagster.io/concepts/configuration/config-schema#specifying-runtime-configuration
resources:
 wandb_config:
   config:
     entity: my_entity # 본인의 W&B entity로 수정
     project: my_project # 본인의 W&B project로 수정
ops:
 my_launched_job:
   config:
     entry_point:
       - python
       - train.py
     queue: my_dagster_queue
     uri: https://github.com/wandb/example-dagster-integration-with-launch


from dagster_wandb.launch.ops import run_launch_job
from dagster_wandb.resources import wandb_resource

from dagster import job, make_values_resource


@job(resource_defs={
       "wandb_config": make_values_resource(
           entity=str,
           project=str,
       ),
       "wandb_resource": wandb_resource.configured(
           {"api_key": {"env": "WANDB_API_KEY"}}
       ),
   },
)
def run_launch_job_example():
   run_launch_job.alias("my_launched_job")() # alias로 job명 변경
```

## 모범 사례(Best practices)

1. Artifacts 읽기/쓰기 시 IO Manager 활용
[`Artifact.download()`]({{< relref path="/ref/python/sdk/classes/artifact#download" lang="ko" >}})나 [`Run.log_artifact()`]({{< relref path="/ref/python/sdk/classes/run#log_artifact" lang="ko" >}}) 등을 직접 호출하는 것보다 integration에서 반환 값만 넘겨주면 나머지 처리는 통합에 맡기는 것이 lineage 및 추적에 유리합니다.

2. 복잡한 케이스 외에는 직접 Artifact 오브젝트를 만들지 않기
일반적인 경우에는 Python 오브젝트나 W&B 오브젝트를 그대로 반환하세요. 통합이 Artifact 생성 및 보강을 알아서 처리합니다. 복잡한 경우에만 직접 Artifact 오브젝트를 만들어 integration에 전달하면, 소스, 버전, 프로토콜 등 메타데이터가 자동으로 보강됩니다.

3. 파일, 디렉토리, 외부 참조 추가는 metadata 활용
`wandb_artifact_configuration`의 설정을 통해 로컬 파일/디렉토리, 외부 참조(S3, GCS, HTTP 등) 등을 쉽게 추가할 수 있습니다. 상세 예시는 [Artifact 설정]({{< relref path="#configuration-1" lang="ko" >}})을 참고하세요.

4. Artifact를 생성할 때는 @op 보다는 @asset 사용 권장
Artifact는 assets입니다. Dagster에서 asset으로 관리할 때 @asset을 활용하는 것이 관측성·추적성 측면에서 이점이 있습니다.

5. Dagster 외부에서 생성한 Artifact 사용은 SourceAsset 활용
외부에서 생성된 Artifact도 SourceAsset을 설정하면 통합 기능을 그대로 사용할 수 있습니다.

6. 대형 모델 트레이닝 오케스트레이션에는 W&B Launch 활용 권장
소형 모델은 Dagster 클러스터 내에서, 대형 모델은 W&B Launch를 이용해 클러스터 리소스 과부하 없이 효율적으로 트레이닝하세요. GPU/전용 컴퓨터 자원 활용도 간편해집니다.

7. Dagster에서 experiment tracking 시 W&B Run ID를 Dagster Run ID로 세팅
[Run 재개]({{< relref path="/guides/models/track/runs/resuming.md" lang="ko" >}}) 설정과 함께, W&B Run ID를 Dagster Run ID(또는 직접 지정한 문자열)로 맞추세요. 이렇게 하면 W&B Metrics와 Artifacts가 동일 W&B Run 내에 저장되어 실험 이력 관리가 쉬워집니다.

W&B Run ID를 Dagster Run ID로 지정하는 예:
```python
wandb.init(
    id=context.run_id,
    resume="allow",
    ...
)
```

직접 지정한 W&B Run ID를 IO Manager 설정에 넘기는 예:
```python
wandb.init(
    id="my_resumable_run_id",
    resume="allow",
    ...
)

@job(
   resource_defs={
       "io_manager": wandb_artifacts_io_manager.configured(
           {"wandb_run_id": "my_resumable_run_id"}
       ),
   }
)
```

8. 대용량 W&B Artifact에서는 get 또는 get_path로 필요한 데이터만 선택적으로 수집
기본적으로 Integration은 Artifact 전체를 다운로드합니다. 대용량인 경우 필요한 파일/오브젝트만 get 또는 get_path로 부분 수집하면 성능과 자원 사용을 개선할 수 있습니다.

9. Python 오브젝트의 경우 유스 케이스에 따라 직렬화 모듈을 조정
기본으로는 [pickle](https://docs.python.org/3/library/pickle.html) 모듈이 사용되며, yield가 있는 함수 등 일부는 pickle에서 오류가 발생할 수 있습니다. W&B는 [dill](https://github.com/uqfoundation/dill), [cloudpickle](https://github.com/cloudpipe/cloudpickle), [joblib](https://github.com/joblib/joblib) 등도 지원합니다.

필요하다면 [ONNX](https://onnx.ai/)나 [PMML](https://en.wikipedia.org/wiki/Predictive_Model_Markup_Language)처럼 더 진보된 직렬화도 가능합니다. 유스 케이스별 적합한 방식은 관련 자료를 참고하세요.
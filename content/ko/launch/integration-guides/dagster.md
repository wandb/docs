---
title: Dagster
description: W&B를 Dagster와 통합하는 방법에 대한 가이드 입니다.
menu:
  launch:
    identifier: ko-launch-integration-guides-dagster
    parent: launch-integration-guides
url: guides/integrations/dagster
---

Dagster와 W&B를 사용하여 MLOps 파이프라인을 조율하고 ML 자산을 관리하세요. Weights & Biases와의 통합을 통해 Dagster 내에서 다음을 쉽게 수행할 수 있습니다.

* [W&B Artifacts]({{< relref path="/guides/core/artifacts/" lang="ko" >}})를 사용하고 생성합니다.
* [W&B Model Registry]({{< relref path="/guides/models/registry/model_registry/" lang="ko" >}})에서 Registered Models를 사용하고 생성합니다.
* [W&B Launch]({{< relref path="/launch/" lang="ko" >}})를 사용하여 전용 컴퓨팅에서 트레이닝 작업을 실행합니다.
* ops 및 assets에서 [wandb]({{< relref path="/ref/python/" lang="ko" >}}) 클라이언트를 사용합니다.

W&B Dagster 통합은 W&B 관련 Dagster 리소스 및 IO 관리자를 제공합니다.

* `wandb_resource`: W&B API에 인증하고 통신하는 데 사용되는 Dagster 리소스입니다.
* `wandb_artifacts_io_manager`: W&B Artifacts를 사용하는 데 사용되는 Dagster IO 관리자입니다.

다음 가이드에서는 Dagster에서 W&B를 사용하기 위한 필수 조건을 충족하는 방법, ops 및 assets에서 W&B Artifacts를 생성하고 사용하는 방법, W&B Launch 사용 방법 및 권장 모범 사례를 보여줍니다.

## 시작하기 전에
Weights & Biases 내에서 Dagster를 사용하려면 다음 리소스가 필요합니다.
1. **W&B API 키**.
2. **W&B 엔터티(사용자 또는 팀)**: 엔터티는 W&B Runs 및 Artifacts를 보내는 사용자 이름 또는 팀 이름입니다. Runs를 기록하기 전에 W&B App UI에서 계정 또는 팀 엔터티를 생성해야 합니다. 엔터티를 지정하지 않으면 Run은 기본 엔터티(일반적으로 사용자 이름)로 전송됩니다. **프로젝트 기본값**에서 설정에서 기본 엔터티를 변경합니다.
3. **W&B 프로젝트**: [W&B Runs]({{< relref path="/guides/models/track/runs/" lang="ko" >}})가 저장되는 프로젝트의 이름입니다.

W&B App에서 해당 사용자 또는 팀의 프로필 페이지를 확인하여 W&B 엔터티를 찾습니다. 기존 W&B 프로젝트를 사용하거나 새 프로젝트를 만들 수 있습니다. 새 프로젝트는 W&B App 홈페이지 또는 사용자/팀 프로필 페이지에서 만들 수 있습니다. 프로젝트가 존재하지 않으면 처음 사용할 때 자동으로 생성됩니다. 다음 지침은 API 키를 얻는 방법을 보여줍니다.

### API 키를 얻는 방법
1. [W&B에 로그인](https://wandb.ai/login)합니다. 참고: W&B Server를 사용하는 경우 관리자에게 인스턴스 호스트 이름을 문의하세요.
2. [인증 페이지](https://wandb.ai/authorize) 또는 사용자/팀 설정에서 API 키를 수집합니다. 프로덕션 환경의 경우 [서비스 계정]({{< relref path="../../support/service_account_useful.md" lang="ko" >}})을 사용하여 해당 키를 소유하는 것이 좋습니다.
3. 해당 API 키 내보내기 `WANDB_API_KEY=YOUR_KEY`에 대한 환경 변수를 설정합니다.

다음 예제에서는 Dagster 코드에서 API 키를 지정할 위치를 보여줍니다. `wandb_config` 중첩 사전 내에서 엔터티 및 프로젝트 이름을 지정해야 합니다. 다른 W&B 프로젝트를 사용하려면 다른 `wandb_config` 값을 다른 ops/assets에 전달할 수 있습니다. 전달할 수 있는 가능한 키에 대한 자세한 내용은 아래의 구성 섹션을 참조하세요.

{{< tabpane text=true >}}
{{% tab "Config for @job" %}}
예: `@job`에 대한 구성
```python
# config.yaml에 다음을 추가합니다.
# 또는 Dagit의 Launchpad 또는 JobDefinition.execute_in_process에서 구성을 설정할 수 있습니다.
# 참조: https://docs.dagster.io/concepts/configuration/config-schema#specifying-runtime-configuration
resources:
 wandb_config:
   config:
     entity: my_entity # 이를 W&B 엔터티로 바꿉니다.
     project: my_project # 이를 W&B 프로젝트로 바꿉니다.


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

예: assets를 사용하는 `@repository`에 대한 구성

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
                   {"cache_duration_in_minutes": 60} # 파일을 1시간 동안만 캐시합니다.
               ),
           },
           resource_config_by_key={
               "wandb_config": {
                   "config": {
                       "entity": "my_entity", # 이를 W&B 엔터티로 바꿉니다.
                       "project": "my_project", # 이를 W&B 프로젝트로 바꿉니다.
                   }
               }
           },
       ),
   ]
```
`@job`에 대한 예제와 달리 이 예제에서는 IO 관리자 캐시 기간을 구성하고 있습니다.
{{% /tab %}}
{{< /tabpane >}}

### 구성
다음 구성 옵션은 통합에서 제공하는 W&B 관련 Dagster 리소스 및 IO 관리자의 설정으로 사용됩니다.

* `wandb_resource`: W&B API와 통신하는 데 사용되는 Dagster [리소스](https://docs.dagster.io/concepts/resources)입니다. 제공된 API 키를 사용하여 자동으로 인증합니다. 속성:
    * `api_key`: (str, 필수): W&B API와 통신하는 데 필요한 W&B API 키입니다.
    * `host`: (str, 선택 사항): 사용할 API 호스트 서버입니다. W&B Server를 사용하는 경우에만 필요합니다. 기본값은 퍼블릭 클라우드 호스트인 `https://api.wandb.ai`입니다.
* `wandb_artifacts_io_manager`: W&B Artifacts를 사용하는 Dagster [IO 관리자](https://docs.dagster.io/concepts/io-management/io-managers)입니다. 속성:
    * `base_dir`: (int, 선택 사항) 로컬 스토리지 및 캐싱에 사용되는 기본 디렉터리입니다. W&B Artifacts 및 W&B Run 로그는 해당 디렉터리에 기록되고 해당 디렉터리에서 읽습니다. 기본적으로 `DAGSTER_HOME` 디렉터리를 사용합니다.
    * `cache_duration_in_minutes`: (int, 선택 사항) W&B Artifacts 및 W&B Run 로그를 로컬 스토리지에 보관해야 하는 시간(분)을 정의합니다. 해당 시간 동안 열리지 않은 파일 및 디렉터리만 캐시에서 제거됩니다. 캐시 제거는 IO 관리자 실행이 끝날 때 발생합니다. 캐싱을 완전히 끄려면 0으로 설정할 수 있습니다. 캐싱은 Artifact가 동일한 시스템에서 실행되는 작업 간에 재사용될 때 속도를 향상시킵니다. 기본값은 30일입니다.
    * `run_id`: (str, 선택 사항): 재개에 사용되는 이 Run에 대한 고유 ID입니다. 프로젝트에서 고유해야 하며, Run을 삭제하면 ID를 재사용할 수 없습니다. 짧은 설명 이름에는 이름 필드를 사용하고, Run 간에 비교하기 위해 하이퍼파라미터를 저장하는 데는 구성을 사용합니다. ID에 다음 특수 문자를 포함할 수 없습니다. `/\#?%:..` IO 관리자가 Run을 재개할 수 있도록 Dagster 내에서 실험 추적을 수행할 때 Run ID를 설정해야 합니다. 기본적으로 Dagster Run ID(예: `7e4df022-1bf2-44b5-a383-bb852df4077e`)로 설정됩니다.
    * `run_name`: (str, 선택 사항) UI에서 이 Run을 식별하는 데 도움이 되는 이 Run에 대한 짧은 표시 이름입니다. 기본적으로 `dagster-run-[Dagster Run ID의 처음 8자]` 형식의 문자열입니다. 예를 들어 `dagster-run-7e4df022`입니다.
    * `run_tags`: (list[str], 선택 사항): UI에서 이 Run에 태그 목록을 채우는 문자열 목록입니다. 태그는 Run을 함께 구성하거나 `베이스라인` 또는 `프로덕션`과 같은 임시 레이블을 적용하는 데 유용합니다. UI에서 태그를 쉽게 추가 및 제거하거나 특정 태그가 있는 Run으로 필터링할 수 있습니다. 통합에서 사용하는 모든 W&B Run에는 `dagster_wandb` 태그가 있습니다.

## W&B Artifacts 사용

W&B Artifact와의 통합은 Dagster IO 관리자를 기반으로 합니다.

[IO 관리자](https://docs.dagster.io/concepts/io-management/io-managers)는 asset 또는 op의 출력을 저장하고 다운스트림 assets 또는 ops에 대한 입력으로 로드하는 역할을 담당하는 사용자 제공 오브젝트입니다. 예를 들어 IO 관리자는 파일 시스템의 파일에서 오브젝트를 저장하고 로드할 수 있습니다.

통합은 W&B Artifacts에 대한 IO 관리자를 제공합니다. 이를 통해 모든 Dagster `@op` 또는 `@asset`이 W&B Artifacts를 기본적으로 생성하고 사용할 수 있습니다. 다음은 Python 목록을 포함하는 dataset 유형의 W&B Artifact를 생성하는 `@asset`의 간단한 예입니다.

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
    return [1, 2, 3] # 이는 Artifact에 저장됩니다.
```

Artifacts를 쓰기 위해 메타데이터 구성으로 `@op`, `@asset` 및 `@multi_asset`에 주석을 달 수 있습니다. 마찬가지로 Dagster 외부에서 생성된 경우에도 W&B Artifacts를 사용할 수 있습니다.

## W&B Artifacts 쓰기
계속하기 전에 W&B Artifacts를 사용하는 방법에 대한 이해가 필요합니다. [Artifacts 가이드]({{< relref path="/guides/core/artifacts/" lang="ko" >}})를 읽어보세요.

Python 함수에서 오브젝트를 반환하여 W&B Artifact를 씁니다. W&B는 다음 오브젝트를 지원합니다.
* Python 오브젝트(int, dict, list…)
* W&B 오브젝트(Table, Image, Graph…)
* W&B Artifact 오브젝트

다음 예제에서는 Dagster assets(`@asset`)로 W&B Artifacts를 쓰는 방법을 보여줍니다.

{{< tabpane text=true >}}
{{% tab "Python objects" %}}
[pickle](https://docs.python.org/3/library/pickle.html) 모듈로 직렬화할 수 있는 모든 항목은 피클링되어 통합에서 생성된 Artifact에 추가됩니다. 콘텐츠는 Dagster 내부에서 해당 Artifact를 읽을 때 unpickled됩니다(자세한 내용은 [Artifact 읽기]({{< relref path="#read-wb-artifacts" lang="ko" >}}) 참조).

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

W&B는 여러 Pickle 기반 직렬화 모듈([pickle](https://docs.python.org/3/library/pickle.html), [dill](https://github.com/uqfoundation/dill), [cloudpickle](https://github.com/cloudpipe/cloudpickle), [joblib](https://github.com/joblib/joblib))을 지원합니다. [ONNX](https://onnx.ai/) 또는 [PMML](https://en.wikipedia.org/wiki/Predictive_Model_Markup_Language)과 같은 고급 직렬화를 사용할 수도 있습니다. 자세한 내용은 [직렬화]({{< relref path="#serialization-configuration" lang="ko" >}}) 섹션을 참조하세요.
{{% /tab %}}
{{% tab "W&B Object" %}}
모든 기본 W&B 오브젝트([Table]({{< relref path="/ref/python/data-types/table.md" lang="ko" >}}), [Image]({{< relref path="/ref/python/data-types/image.md" lang="ko" >}}) 또는 [Graph]({{< relref path="/ref/python/data-types/graph.md" lang="ko" >}}))는 통합에서 생성된 Artifact에 추가됩니다. 다음은 Table을 사용하는 예입니다.

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

복잡한 유스 케이스의 경우 Artifact 오브젝트를 직접 빌드해야 할 수 있습니다. 통합은 여전히 통합 양쪽에서 메타데이터를 보강하는 것과 같은 유용한 추가 기능을 제공합니다.

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

### 구성
`wandb_artifact_configuration`이라는 구성 사전은 `@op`, `@asset` 및 `@multi_asset`에서 설정할 수 있습니다. 이 사전은 데코레이터 인수로 메타데이터로 전달해야 합니다. 이 구성은 W&B Artifacts의 IO 관리자 읽기 및 쓰기를 제어하는 데 필요합니다.

`@op`의 경우 [Out](https://docs.dagster.io/_apidocs/ops#dagster.Out) 메타데이터 인수를 통해 출력 메타데이터에 있습니다.
`@asset`의 경우 asset의 메타데이터 인수에 있습니다.
`@multi_asset`의 경우 [AssetOut](https://docs.dagster.io/_apidocs/assets#dagster.AssetOut) 메타데이터 인수를 통해 각 출력 메타데이터에 있습니다.

다음 코드 예제에서는 `@op`, `@asset` 및 `@multi_asset` 계산에서 사전을 구성하는 방법을 보여줍니다.

{{< tabpane text=true >}}
{{% tab "Example for @op" %}}
`@op`에 대한 예:
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
`@asset`에 대한 예:
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

@asset에는 이미 이름이 있으므로 구성을 통해 이름을 전달할 필요가 없습니다. 통합은 Artifact 이름을 asset 이름으로 설정합니다.

{{% /tab %}}
{{% tab "Example for @multi_asset" %}}

`@multi_asset`에 대한 예:

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

지원되는 속성:
* `name`: (str) 이 Artifact에 대한 사람이 읽을 수 있는 이름입니다. UI에서 이 Artifact를 식별하거나 use_artifact 호출에서 참조할 수 있는 방법입니다. 이름에는 문자, 숫자, 밑줄, 하이픈 및 마침표가 포함될 수 있습니다. 이름은 프로젝트 전체에서 고유해야 합니다. `@op`에 필요합니다.
* `type`: (str) Artifact의 유형입니다. Artifact를 구성하고 구별하는 데 사용됩니다. 일반적인 유형에는 데이터 세트 또는 모델이 있지만 문자, 숫자, 밑줄, 하이픈 및 마침표를 포함하는 모든 문자열을 사용할 수 있습니다. 출력이 Artifact가 아닌 경우에 필요합니다.
* `description`: (str) Artifact에 대한 설명을 제공하는 자유 텍스트입니다. 설명은 UI에서 렌더링된 마크다운이므로 테이블, 링크 등을 배치하기에 좋은 장소입니다.
* `aliases`: (list[str]) Artifact에 적용할 하나 이상의 에일리어스를 포함하는 배열입니다. 통합은 설정 여부에 관계없이 해당 목록에 "latest" 태그도 추가합니다. 이는 모델 및 데이터 세트의 버전 관리를 관리하는 효과적인 방법입니다.
* [`add_dirs`]({{< relref path="/ref/python/artifact.md#add_dir" lang="ko" >}}): (list[dict[str, Any]]): Artifact에 포함할 각 로컬 디렉터리에 대한 구성을 포함하는 배열입니다. SDK의 동명 메서드와 동일한 인수를 지원합니다.
* [`add_files`]({{< relref path="/ref/python/artifact.md#add_file" lang="ko" >}}): (list[dict[str, Any]]): Artifact에 포함할 각 로컬 파일에 대한 구성을 포함하는 배열입니다. SDK의 동명 메서드와 동일한 인수를 지원합니다.
* [`add_references`]({{< relref path="/ref/python/artifact.md#add_reference" lang="ko" >}}): (list[dict[str, Any]]): Artifact에 포함할 각 외부 참조에 대한 구성을 포함하는 배열입니다. SDK의 동명 메서드와 동일한 인수를 지원합니다.
* `serialization_module`: (dict) 사용할 직렬화 모듈의 구성입니다. 자세한 내용은 직렬화 섹션을 참조하세요.
    * `name`: (str) 직렬화 모듈의 이름입니다. 허용되는 값: `pickle`, `dill`, `cloudpickle`, `joblib`. 모듈은 로컬에서 사용할 수 있어야 합니다.
    * `parameters`: (dict[str, Any]) 직렬화 함수에 전달되는 선택적 인수입니다. 해당 모듈에 대한 dump 메서드와 동일한 파라미터를 허용합니다. 예를 들어 `{"compress": 3, "protocol": 4}`입니다.

고급 예:

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

asset은 통합 양쪽에서 유용한 메타데이터로 구체화됩니다.
* W&B 쪽: 소스 통합 이름 및 버전, 사용된 Python 버전, pickle 프로토콜 버전 등입니다.
* Dagster 쪽:
    * Dagster Run ID
    * W&B Run: ID, 이름, 경로, URL
    * W&B Artifact: ID, 이름, 유형, 버전, 크기, URL
    * W&B 엔터티
    * W&B 프로젝트

다음 이미지는 Dagster asset에 추가된 W&B의 메타데이터를 보여줍니다. 이 정보는 통합 없이는 사용할 수 없습니다.

{{< img src="/images/integrations/dagster_wb_metadata.png" alt="" >}}

다음 이미지는 제공된 구성이 W&B Artifact의 유용한 메타데이터로 보강된 방법을 보여줍니다. 이 정보는 재현성 및 유지 관리에 도움이 되어야 합니다. 통합 없이는 사용할 수 없습니다.

{{< img src="/images/integrations/dagster_inte_1.png" alt="" >}}
{{< img src="/images/integrations/dagster_inte_2.png" alt="" >}}
{{< img src="/images/integrations/dagster_inte_3.png" alt="" >}}

{{% alert %}}
mypy와 같은 정적 유형 검사기를 사용하는 경우 다음을 사용하여 구성 유형 정의 오브젝트를 가져옵니다.

```python
from dagster_wandb import WandbArtifactConfiguration
```
{{% /alert %}}

### 파티션 사용

통합은 기본적으로 [Dagster 파티션](https://docs.dagster.io/concepts/partitions-schedules-sensors/partitions)을 지원합니다.

다음은 `DailyPartitionsDefinition`을 사용하여 분할된 예입니다.
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
이 코드는 각 파티션에 대해 하나의 W&B Artifact를 생성합니다. 파티션 키가 추가된 asset 이름 아래의 Artifact 패널(UI)에서 Artifacts를 봅니다. 예를 들어 `my_daily_partitioned_asset.2023-01-01`, `my_daily_partitioned_asset.2023-01-02` 또는 `my_daily_partitioned_asset.2023-01-03`입니다. 여러 차원에서 분할된 Assets는 각 차원을 점으로 구분된 형식으로 표시합니다. 예를 들어 `my_asset.car.blue`입니다.

{{% alert color="secondary" %}}
통합은 한 번의 Run 내에서 여러 파티션의 구체화를 허용하지 않습니다. Assets를 구체화하려면 여러 번의 Run을 수행해야 합니다. 이는 Assets를 구체화할 때 Dagit에서 실행할 수 있습니다.

{{< img src="/images/integrations/dagster_multiple_runs.png" alt="" >}}
{{% /alert %}}

#### 고급 사용법
- [분할된 작업](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/ops/partitioned_job.py)
- [단순 분할된 asset](https://github.com/wandb/dagster/blob/master/examples/with_wandb/with_wandb/assets/simple_partitions_example.py)
- [다중 분할된 asset](https://github.com/wandb/dagster/blob/master/examples/with_wandb/with_wandb/assets/multi_partitions_example.py)
- [고급 분할된 사용법](https://github.com/wandb/dagster/blob/master/examples/with_wandb/with_wandb/assets/advanced_partitions_example.py)

## W&B Artifacts 읽기
W&B Artifacts 읽기는 쓰기와 유사합니다. `wandb_artifact_configuration`이라는 구성 사전은 `@op` 또는 `@asset`에서 설정할 수 있습니다. 유일한 차이점은 출력이 아닌 입력에서 구성을 설정해야 한다는 것입니다.

`@op`의 경우 [In](https://docs.dagster.io/_apidocs/ops#dagster.In) 메타데이터 인수를 통해 입력 메타데이터에 있습니다. Artifact 이름을 명시적으로 전달해야 합니다.

`@asset`의 경우 [Asset](https://docs.dagster.io/_apidocs/assets#dagster.AssetIn) In 메타데이터 인수를 통해 입력 메타데이터에 있습니다. 상위 asset의 이름과 일치해야 하므로 Artifact 이름을 전달해서는 안 됩니다.

통합 외부에서 생성된 Artifact에 대한 종속성이 있는 경우 [SourceAsset](https://docs.dagster.io/_apidocs/assets#dagster.SourceAsset)을 사용해야 합니다. 항상 해당 asset의 최신 버전을 읽습니다.

다음 예제에서는 다양한 ops에서 Artifact를 읽는 방법을 보여줍니다.

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
다른 `@asset`에서 생성한 Artifact 읽기
```python
@asset(
   name="my_asset",
   ins={
       "artifact": AssetIn(
           # 입력 인수의 이름을 바꾸지 않으려면 'key'를 제거할 수 있습니다.
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

Dagster 외부에서 생성된 Artifact 읽기:

```python
my_artifact = SourceAsset(
   key=AssetKey("my_artifact"),  # W&B Artifact의 이름
   description="Dagster 외부에서 생성된 Artifact",
   io_manager_key="wandb_artifacts_manager",
)


@asset
def read_artifact(context, my_artifact):
   context.log.info(my_artifact)
```
{{% /tab %}}
{{< /tabpane >}}

### 구성
다음 구성은 데코레이터된 함수에 대한 입력으로 IO 관리자가 수집하고 제공해야 하는 항목을 나타내는 데 사용됩니다. 다음과 같은 읽기 패턴이 지원됩니다.

1. Artifact 내에 포함된 명명된 오브젝트를 가져오려면 get을 사용합니다.

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

2. Artifact 내에 포함된 다운로드된 파일의 로컬 경로를 가져오려면 get_path를 사용합니다.

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

3. 전체 Artifact 오브젝트를 가져오려면(콘텐츠가 로컬에 다운로드됨):
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

지원되는 속성
* `get`: (str) Artifact 상대 이름에 있는 W&B 오브젝트를 가져옵니다.
* `get_path`: (str) Artifact 상대 이름에 있는 파일의 경로를 가져옵니다.

### 직렬화 구성
기본적으로 통합은 표준 [pickle](https://docs.python.org/3/library/pickle.html) 모듈을 사용하지만 일부 오브젝트는 호환되지 않습니다. 예를 들어 yield가 있는 함수는 피클링하려고 하면 오류가 발생합니다.

더 많은 Pickle 기반 직렬화 모듈([dill](https://github.com/uqfoundation/dill), [cloudpickle](https://github.com/cloudpipe/cloudpickle), [joblib](https://github.com/joblib/joblib))을 지원합니다. 직렬화된 문자열을 반환하거나 Artifact를 직접 만들어 [ONNX](https://onnx.ai/) 또는 [PMML](https://en.wikipedia.org/wiki/Predictive_Model_Markup_Language)과 같은 고급 직렬화를 사용할 수도 있습니다. 올바른 선택은 유스 케이스에 따라 달라지므로 이 주제에 대한 사용 가능한 문헌을 참조하세요.

### Pickle 기반 직렬화 모듈

{{% alert color="secondary" %}}
피클링은 안전하지 않은 것으로 알려져 있습니다. 보안이 우려되는 경우 W&B 오브젝트만 사용하세요. 데이터를 서명하고 해시 키를 자체 시스템에 저장하는 것이 좋습니다. 더 복잡한 유스 케이스의 경우 주저하지 말고 문의해 주세요. 기꺼이 도와드리겠습니다.
{{% /alert %}}

`wandb_artifact_configuration`의 `serialization_module` 사전을 통해 사용된 직렬화를 구성할 수 있습니다. 모듈이 Dagster를 실행하는 시스템에서 사용할 수 있는지 확인하세요.

통합은 해당 Artifact를 읽을 때 사용할 직렬화 모듈을 자동으로 알 수 있습니다.

현재 지원되는 모듈은 `pickle`, `dill`, `cloudpickle` 및 `joblib`입니다.

다음은 joblib로 직렬화된 "모델"을 만들고 추론에 사용하는 단순화된 예입니다.

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
    # 이것은 실제 ML 모델이 아니지만 pickle 모듈로는 불가능합니다.
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
    context.log.info(inference_result)  # Prints: 3
    return inference_result
```

### 고급 직렬화 형식(ONNX, PMML)
ONNX 및 PMML과 같은 교환 파일 형식을 사용하는 것이 일반적입니다. 통합은 이러한 형식을 지원하지만 Pickle 기반 직렬화보다 더 많은 작업이 필요합니다.

이러한 형식을 사용하는 두 가지 다른 방법이 있습니다.
1. 모델을 선택한 형식으로 변환한 다음 해당 형식의 문자열 표현을 일반 Python 오브젝트인 것처럼 반환합니다. 통합은 해당 문자열을 피클링합니다. 그런 다음 해당 문자열을 사용하여 모델을 다시 빌드할 수 있습니다.
2. 직렬화된 모델이 있는 새 로컬 파일을 만든 다음 add_file 구성을 사용하여 해당 파일로 사용자 지정 Artifact를 빌드합니다.

다음은 ONNX를 사용하여 직렬화되는 Scikit-learn 모델의 예입니다.

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
    # Inspired from https://onnx.ai/sklearn-onnx/

    # Train a model.
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clr = RandomForestClassifier()
    clr.fit(X_train, y_train)

    # Convert into ONNX format
    initial_type = [("float_input", FloatTensorType([None, 4]))]
    onx = convert_sklearn(clr, initial_types=initial_type)

    # Write artifacts (model + test_set)
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
    # Inspired from https://onnx.ai/sklearn-onnx/

    # Compute the prediction with ONNX Runtime
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

통합은 기본적으로 [Dagster 파티션](https://docs.dagster.io/concepts/partitions-schedules-sensors/partitions)을 지원합니다.

asset의 하나, 여러 개 또는 모든 파티션을 선택적으로 읽을 수 있습니다.

모든 파티션은 파티션 키와 Artifact 콘텐츠를 각각 나타내는 키와 값으로 사전에 제공됩니다.

{{< tabpane text=true >}}
{{% tab "Read all partitions" %}}
사전으로 제공되는 업스트림 `@asset`의 모든 파티션을 읽습니다. 이 사전에서 키와 값은 각각 파티션 키와 Artifact 콘텐츠에 해당합니다.
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
`AssetIn`의 `partition_mapping` 구성을 사용하면 특정 파티션을 선택할 수 있습니다. 이 경우 `TimeWindowPartitionMapping`을 사용하고 있습니다.
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

구성 오브젝트인 `metadata`는 프로젝트에서 Weights & Biases(wandb)가 다양한 Artifact 파티션과 상호 작용하는 방식을 구성하는 데 사용됩니다.

오브젝트 `metadata`에는 중첩된 오브젝트인 `partitions`를 추가로 포함하는 `wandb_artifact_configuration`이라는 키가 포함되어 있습니다.

`partitions` 오브젝트는 각 파티션의 이름을 해당 구성에 매핑합니다. 각 파티션에 대한 구성은 데이터를 검색하는 방법을 지정할 수 있습니다. 이러한 구성에는 각 파티션의 요구 사항에 따라 `get`, `version` 및 `alias`라는 다양한 키가 포함될 수 있습니다.

**구성 키**

1. `get`:
`get` 키는 데이터를 가져올 W&B 오브젝트(Table, Image...)의 이름을 지정합니다.
2. `version`:
`version` 키는 Artifact에 대한 특정 버전을 가져오려는 경우에 사용됩니다.
3. `alias`:
`alias` 키를 사용하면 에일리어스로 Artifact를 가져올 수 있습니다.

**와일드카드 구성**

와일드카드 `"*"`는 구성되지 않은 모든 파티션을 나타냅니다. 이는 `partitions` 오브젝트에서 명시적으로 언급되지 않은 파티션에 대한 기본 구성을 제공합니다.

예를 들어,

```python
"*": {
    "get": "default_table_name",
},
```
이 구성은 명시적으로 구성되지 않은 모든 파티션에 대해 데이터가 `default_table_name`이라는 테이블에서 가져옴을 의미합니다.

**특정 파티션 구성**

키를 사용하여 특정 구성을 제공하여 특정 파티션에 대한 와일드카드 구성을 재정의할 수 있습니다.

예를 들어,

```python
"yellow": {
    "get": "custom_table_name",
},
```

이 구성은 `yellow`라는 파티션의 경우 데이터가 와일드카드 구성을 재정의하여 `custom_table_name`이라는 테이블에서 가져옴을 의미합니다.

**버전 관리 및 에일리어싱**

버전 관리 및 에일리어싱을 위해 구성에서 특정 `version` 및 `alias` 키를 제공할 수 있습니다.

버전의 경우,

```python
"orange": {
    "version": "v0",
},
```

이 구성은 `orange` Artifact 파티션의 버전 `v0`에서 데이터를 가져옵니다.

에일리어스의 경우,

```python
"blue": {
    "alias": "special_alias",
},
```

이 구성은 에일리어스 `special_alias`(구성에서 `blue`로 지칭)가 있는 Artifact 파티션의 테이블 `default_table_name`에서 데이터를 가져옵니다.

### 고급 사용법
통합의 고급 사용법을 보려면 다음 전체 코드 예제를 참조하세요.
* [Assets에 대한 고급 사용법 예제](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/assets/advanced_example.py)
* [분할된 작업 예제](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/ops/partitioned_job.py)
* [모델을 Model Registry에 연결](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/assets/model_registry_example.py)

## W&B Launch 사용

{{% alert color="secondary" %}}
활성 개발 중인 베타 제품
Launch에 관심이 있으신가요? 계정 팀에 문의하여 W&B Launch에 대한 고객 파일럿 프로그램에 참여하는 것에 대해 논의하세요.
파일럿 고객은 AWS EKS 또는 SageMaker를 사용하여 베타 프로그램에 참여해야 합니다. 궁극적으로는 추가 플랫폼을 지원할 계획입니다.
{{% /alert %}}

계속하기 전에 W&B Launch를 사용하는 방법에 대한 이해가 필요합니다. Launch 가이드: /guides/launch를 읽어보세요.

Dagster 통합은 다음을 지원합니다.
* Dagster 인스턴스에서 하나 또는 여러 개의 Launch 에이전트 실행.
* Dagster 인스턴스 내에서 로컬 Launch 작업 실행.
* 온프레미스 또는 클라우드에서 원격 Launch 작업 실행.

### Launch 에이전트
통합은 `run_launch_agent`라는 가져올 수 있는 `@op`을 제공합니다. Launch 에이전트를 시작하고 수동으로 중지할 때까지 장기 실행 프로세스로 실행합니다.

에이전트는 Launch 대기열을 폴링하고 작업을 실행(또는 실행될 외부 서비스로 디스패치)하는 프로세스입니다.

구성에 대한 [참조 문서]({{< relref path="/launch/" lang="ko" >}})를 참조하세요.

Launchpad에서 모든 속성에 대한 유용한 설명을 볼 수도 있습니다.

{{< img src="/images/integrations/dagster_launch_agents.png" alt="" >}}

간단한 예
```python
# config.yaml에 다음을 추가합니다.
# 또는 Dagit의 Launchpad 또는 JobDefinition.execute_in_process에서 구성을 설정할 수 있습니다.
# 참조: https://docs.dagster.io/concepts/configuration/config-schema#specifying-runtime-configuration
resources:
 wandb_config:
   config:
     entity: my_entity # 이를 W&B 엔터티로 바꿉니다.
     project: my_project # 이를 W&B 프로젝트로 바꿉니다.
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

### Launch 작업
통합은 `run_launch_job`이라는 가져올 수 있는 `@op`을 제공합니다. Launch 작업을 실행합니다.

Launch 작업은 실행하기 위해 대기열에 할당됩니다. 대기열을 만들거나 기본 대기열을 사용할 수 있습니다. 해당 대기열을 수신하는 활성 에이전트가 있는지 확인하세요. Dagster 인스턴스 내에서 에이전트를 실행할 수 있지만 Kubernetes에서 배포 가능한 에이전트를 사용하는 것을 고려할 수도 있습니다.

구성에 대한 [참조 문서]({{< relref path="/launch/" lang="ko" >}})를 참조하세요.

Launchpad에서 모든 속성에 대한 유용한 설명을 볼 수도 있습니다.

{{< img src="/images/integrations/dagster_launch_jobs.png" alt="" >}}

간단한 예
```python
# config.yaml에 다음을 추가합니다.
# 또는 Dagit의 Launchpad 또는 JobDefinition.execute_in_process에서 구성을 설정할 수 있습니다.
# 참조: https://docs.dagster.io/concepts/configuration/config-schema#specifying-runtime-configuration
resources:
 wandb_config:
   config:
     entity: my_entity # 이를 W&B 엔터티로 바꿉니다.
     project: my_project # 이를 W&B 프로젝트로 바꿉니다.
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
   run_launch_job.alias("my_launched_job")() # 에일리어스로 작업 이름을 바꿉니다.
```

## 모범 사례

1. IO 관리자를 사용하여 Artifacts를 읽고 씁니다.
[`Artifact.download()`]({{< relref path="/ref/python/artifact.md#download" lang="ko" >}}) 또는 [`Run.log_artifact()`]({{< relref path="/ref/python/run.md#log_artifact" lang="ko" >}})를 직접 사용할 필요가 없습니다. 이러한 메서드는 통합에서 처리합니다. Artifact에 저장하려는 데이터를 반환하고 통합에서 나머지를 처리하도록 하세요. 이렇게 하면 W&B에서 Artifact에 대한 계보가 향상됩니다.

2. 복잡한 유스 케이스에 대해서만 Artifact 오브젝트를 직접 빌드합니다.
Python 오브젝트 및 W&B 오브젝트는 ops/assets에서 반환되어야 합니다. 통합은 Artifact 번들링을 처리합니다.
복잡한 유스 케이스의 경우 Dagster 작업에서 Artifact를 직접 빌드할 수 있습니다. 소스 통합 이름 및 버전, 사용된 Python 버전, pickle 프로토콜 버전 등과 같은 메타데이터 보강을 위해 Artifact 오브젝트를 통합에 전달하는 것이 좋습니다.

3. 메타데이터를 통해 파일, 디렉터리 및 외부 참조를 Artifacts에 추가합니다.
통합 `wandb_artifact_configuration` 오브젝트를 사용하여 파일, 디렉터리 또는 외부 참조(Amazon S3, GCS, HTTP…)를 추가합니다. 자세한 내용은 [Artifact 구성 섹션]({{< relref path="#configuration-1" lang="ko" >}})의 고급 예제를 참조하세요.

4. Artifact가 생성되면 @op 대신 @asset을 사용합니다.
Artifacts는 assets입니다. Dagster가 해당 asset을 유지 관리하는 경우 asset을 사용하는 것이 좋습니다. 이렇게 하면 Dagit Asset Catalog에서 더 나은 관찰 가능성을 제공합니다.

5. SourceAsset을 사용하여 Dagster 외부에서 생성된 Artifact를 사용합니다.
이를 통해 통합을 활용하여 외부에서 생성된 Artifacts를 읽을 수 있습니다. 그렇지 않으면 통합에서 생성된 Artifacts만 사용할 수 있습니다.

6. W&B Launch를 사용하여 대규모 모델에 대한 전용 컴퓨팅에서 트레이닝을 오케스트레이션합니다.
Dagster 클러스터 내부에서 소규모 모델을 트레이닝할 수 있으며 GPU 노드가 있는 Kubernetes 클러스터에서 Dagster를 실행할 수 있습니다. 대규모 모델 트레이닝에는 W&B Launch를 사용하는 것이 좋습니다. 이렇게 하면 인스턴스에 과부하가 걸리는 것을 방지하고 더 적절한 컴퓨팅에 액세스할 수 있습니다.

7. Dagster 내에서 실험 추적을 수행할 때 W&B Run ID를 Dagster Run ID 값으로 설정합니다.
다음을 모두 수행하는 것이 좋습니다. [Run을 재개 가능하게 만들고]({{< relref path="/guides/models/track/runs/resuming.md" lang="ko" >}}) W&B Run ID를 Dagster Run ID 또는 선택한 문자열로 설정합니다. 이 권장 사항을 따르면 Dagster 내에서 모델을 트레이닝할 때 W&B 메트릭 및 W&B Artifacts가 동일한 W&B Run에 저장됩니다.

W&B Run ID를 Dagster Run ID로 설정합니다.
```python
wandb.init(
    id=context.run_id,
    resume="allow",
    ...
)
```

또는 고유한 W&B Run ID를 선택하고 IO 관리자 구성에 전달합니다.
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

8. 대규모 W&B Artifacts의 경우 get 또는 get_path로 필요한 데이터만 수집합니다.
기본적으로 통합은 전체 Artifact를 다운로드합니다. 매우 큰 Artifacts를 사용하는 경우 필요한 특정 파일 또는 오브젝트만 수집하는 것이 좋습니다. 이렇게 하면 속도와 리소스 활용률이 향상됩니다.

9. Python 오브젝트의 경우 유스 케이스에 맞게 피클링 모듈을 조정합니다.
기본적으로 W&B 통합은 표준 [pickle](https://docs.python.org/3/library/pickle.html) 모듈을 사용합니다. 그러나 일부 오브젝트는 호환되지 않습니다. 예를 들어 yield가 있는 함수는 피클링하려고 하면 오류가 발생합니다. W&B는 다른 Pickle 기반 직렬화 모듈([dill](https://github.com/uqfoundation/dill), [cloudpickle](https://github.com/cloudpipe/cloudpickle), [joblib](https://github.com/joblib/joblib))을 지원합니다.

직렬화된 문자열을 반환하거나 Artifact를 직접 만들어 [ONNX](https://onnx.ai/) 또는 [PMML](https://en.wikipedia.org/wiki/Predictive_Model_Markup_Language)과 같은 고급 직렬화를 사용할 수도 있습니다. 올바른 선택은 유스 케이스에 따라 달라지므로 이 주제에 대한 사용 가능한 문헌을 참조하세요.

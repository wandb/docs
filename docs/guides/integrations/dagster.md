---
title: Dagster
description: W&B와 Dagster를 통합하는 방법에 대한 가이드.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

Dagster와 W&B (Weights & Biases)를 활용하여 MLOps 파이프라인을 오케스트레이션하고 ML 자산을 관리하세요. W&B과의 인테그레이션은 Dagster 내에서 다음과 같은 작업을 용이하게 합니다:

* [W&B Artifacts](../artifacts/intro.md)를 사용하고 생성합니다.
* [W&B Model Registry](../model_registry/intro.md)에서 Registered Models를 사용하고 생성합니다.
* [W&B Launch](../launch/intro.md)를 사용하여 전용 컴퓨팅에서 트레이닝 작업을 실행합니다.
* ops 및 assets에서 [wandb](../../ref/python/README.md) 클라이언트를 사용합니다.

W&B Dagster 인테그레이션은 W&B 전용 Dagster 자원 및 IO 매니저를 제공합니다:

* `wandb_resource`: W&B API에 인증하고 통신하는 데 사용되는 Dagster 자원입니다.
* `wandb_artifacts_io_manager`: W&B Artifacts를 소비하는 데 사용되는 Dagster IO 매니저입니다.

다음 가이드는 Dagster에서 W&B를 사용하기 위한 사전 요구 사항을 만족시키는 방법, ops 및 assets에서 W&B Artifacts를 생성하고 사용하는 방법, W&B Launch를 사용하는 방법 및 추천되는 모범 사례를 설명합니다.

## 시작하기 전에
Weights and Biases 내에서 Dagster를 사용하려면 다음 자원이 필요합니다:
1. **W&B API Key**.
2. **W&B 엔터티 (사용자 또는 팀)**: 엔터티는 W&B Runs 및 Artifacts를 전송하는 사용자명 또는 팀 명입니다. run을 기록하기 전에 W&B 앱 UI에서 계정이나 팀 엔터티를 생성하십시오. 엔터티를 지정하지 않으면 run은 기본 엔터티로 전송됩니다. 이는 보통 사용자의 사용자명입니다. **Project Defaults**의 설정에서 기본 엔터티를 변경할 수 있습니다.
3. **W&B project**: [W&B Runs](../runs/intro.md)가 저장되는 프로젝트의 이름입니다.

W&B 앱에서 해당 사용자 또는 팀의 프로필 페이지를 확인하여 W&B 엔터티를 찾습니다. 기존의 W&B 프로젝트를 사용할 수 있거나 새로 만들 수 있습니다. 새로운 프로젝트는 W&B 앱 홈페이지나 사용자/팀 프로필 페이지에서 만들 수 있습니다. 프로젝트가 존재하지 않으면 처음 사용할 때 자동으로 생성됩니다. API 키를 얻는 방법은 다음 설명에서 확인하십시오:

### API 키 얻는 방법
1. [W&B에 로그인](https://wandb.ai/login)합니다. 참고: W&B Server 사용 시, 관리자에게 인스턴스 호스트명을 요청하십시오.
2. API 키를 [인증 페이지](https://wandb.ai/authorize) 또는 사용자/팀 설정에서 수집합니다. 프로덕션 환경에서는 해당 키를 보유할 [서비스 계정](/guides/technical-faq/general#what-is-a-service-account-and-why-is-it-useful)을 사용할 것을 권장합니다.
3. 그 API 키에 대한 환경 변수를 설정합니다: `WANDB_API_KEY=YOUR_KEY`.

다음 예시는 Dagster 코드에서 API 키를 지정하는 곳을 보여줍니다. 다양한 W&B Project 사용을 원한다면, `wandb_config` 환경 사전 내에서 엔터티와 프로젝트명을 지정하십시오. 전달할 수 있는 가능한 키에 대한 자세한 내용은 아래 설정 섹션을 참조하십시오.

<Tabs
  defaultValue="job"
  values={[
    {label: 'configuration for @job', value: 'job'},
    {label: 'configuration for @repository using assets', value: 'repository'},
  ]}>
  <TabItem value="job">

Example: configuration for `@job`
```python
# config.yaml에 추가하십시오
# 대안으로는 Dagit's Launchpad 또는 JobDefinition.execute_in_process에서 설정할 수 있습니다
# Reference: https://docs.dagster.io/concepts/configuration/config-schema#specifying-runtime-configuration
resources:
 wandb_config:
   config:
     entity: my_entity # 이 부분을 자신의 W&B 엔터티로 교체하세요
     project: my_project # 이 부분을 자신의 W&B 프로젝트로 교체하세요


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

  </TabItem>
  <TabItem value="repository">

Example: configuration for `@repository` using assets

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
                   {"cache_duration_in_minutes": 60} # 파일을 한 시간 동안만 캐시합니다
               ),
           },
           resource_config_by_key={
               "wandb_config": {
                   "config": {
                       "entity": "my_entity", # 이 부분을 자신의 W&B 엔터티로 교체하세요
                       "project": "my_project", # 이 부분을 자신의 W&B 프로젝트로 교체하세요
                   }
               }
           },
       ),
   ]
```
이 예시에서는 `@job` 예시와 달리 IO 매니저 캐시 지속 기간을 설정하고 있습니다.

  </TabItem>
</Tabs>

### Configuration
다음 설정 옵션은 인테그레이션에 의해 제공된 W&B 전용 Dagster 리소스 및 IO 매니저에 대한 설정으로 사용됩니다.

* `wandb_resource`: W&B API와 통신하기 위해 사용되는 Dagster [resource](https://docs.dagster.io/concepts/resources). 제공된 API 키를 사용하여 자동으로 인증합니다. 속성:
   * `api_key`: (str, 필수) W&B API와 통신하는 데 필요한 W&B API 키.
   * `host`: (str, 선택적) 사용하려는 API 호스트 서버. W&B Server를 사용하는 경우에만 필요합니다. 기본값은 Public Cloud 호스트: [https://api.wandb.ai](https://api.wandb.ai)입니다.
* `wandb_artifacts_io_manager`: W&B Artifacts를 소비하기 위한 Dagster [IO Manager](https://docs.dagster.io/concepts/io-management/io-managers). 속성:
   * `base_dir`: (int, 선택적) 로컬 저장 및 캐싱을 위한 기본 디렉토리. W&B Artifacts 및 W&B Run 로그는 그 디렉토리에 기록되고 읽힙니다. 기본적으로는 `DAGSTER_HOME` 디렉토리를 사용합니다.
   * `cache_duration_in_minutes`: (int, 선택적) W&B Artifacts 및 W&B Run 로그가 로컬 저장소에 보관되어야 하는 기간을 정의합니다. 그 시간 동안 열리지 않은 파일과 디렉토리만 캐시에서 제거됩니다. 캐시 정리는 IO 매니저 실행이 끝날 때 발생합니다. 캐싱을 완전히 비활성화하려면 0으로 설정할 수 있습니다. 캐싱은 같은 기계에서 실행되는 작업들 사이에서 아티팩트를 재사용할 때 속도를 빠르게 합니다. 기본값은 30일입니다.
   * `run_id`: (str, 선택적) 재개 시 사용되는 이 run의 고유 ID입니다. 프로젝트 내에서 고유해야 하며, run을 삭제하면 ID를 재사용할 수 없습니다. 이름 필드를 간단한 설명으로 사용하거나, run 간 하이퍼파라미터를 비교하는 데 사용하려면 설정을 사용하세요. ID에는 다음과 같은 특수 문자를 포함할 수 없습니다: `/\#?%:..` Dagster 내부에서 실험 추적을 수행하는 경우에 Run ID를 설정해야 IO 매니저가 run을 재개할 수 있습니다. 기본값은 Dagster Run ID로 설정됩니다. 예: `7e4df022-1bf2-44b5-a383-bb852df4077e`.
   * `run_name`: (str, 선택적) 이 run의 간단한 표시 이름입니다. UI에서 이 run을 식별하는 방법입니다. 기본적으로는 다음 형식의 문자열로 설정됩니다: dagster-run-[Dagster Run ID의 처음 8자] 예: `dagster-run-7e4df022`.
   * `run_tags`: (list[str], 선택적) 이 run의 UI에 태그 목록을 채우는 문자열 목록입니다. 태그는 run을 함께 구성하거나 "baseline" 또는 "production"과 같은 임시 레이블을 적용하는 데 유용합니다. UI에서 태그를 쉽게 추가 및 제거하거나 특정 태그만 있는 run으로 필터링할 수 있습니다. 인테그레이션에서 사용하는 모든 W&B Run은 `dagster_wandb` 태그를 가집니다.

## Use W&B Artifacts

W&B Artifact와의 인테그레이션은 Dagster IO 매니저에 의존합니다.

[IO 매니저](https://docs.dagster.io/concepts/io-management/io-managers)는 자산 또는 op의 출력을 저장하고 다운스트림 자산이나 ops에 입력으로 로드하는 책임을 가진 사용자 제공 오브젝트입니다. 예를 들어, IO 매니저는 파일 시스템의 파일에서 오브젝트를 저장하고 로드할 수 있습니다.

인테그레이션은 W&B Artifacts를 위한 IO 매니저를 제공합니다. 이를 통해 Dagster의 `@op` 또는 `@asset`이 W&B Artifacts를 원활하게 생성하고 소비할 수 있습니다. 여기 `@asset`이 파이썬 리스트를 포함하는 데이터셋 타입의 W&B Artifact를 생성하는 간단한 예제입니다.

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
    return [1, 2, 3] # 이는 아티팩트에 저장됩니다
```

`@op`, `@asset` 및 `@multi_asset`을 메타데이터 설정으로 주석 처리하여 Artifacts를 작성할 수 있습니다. 유사하게 Dagster 외부에서 생성되었더라도 W&B Artifacts를 소비할 수 있습니다.

## Write W&B Artifacts
계속하기 전에 W&B Artifacts 사용법에 대한 충분한 이해를 권장합니다. [Artifacts 가이드](../artifacts/intro.md)를 읽어보세요.

Python 함수에서 객체를 반환하여 W&B Artifact를 작성할 수 있습니다. W&B에서 지원하는 객체는 다음과 같습니다:
* Python 객체 (int, dict, list...)
* W&B 객체 (Table, Image, Graph...)
* W&B Artifact 객체

다음 예제는 Dagster assets (`@asset`)를 통해 W&B Artifacts를 작성하는 방법을 보여줍니다:

<Tabs
  defaultValue="python_objects"
  values={[
    {label: 'Python objects', value: 'python_objects'},
    {label: 'W&B object', value: 'wb_object'},
    {label: 'W&B Artifacts', value: 'wb_artifact'},
  ]}>
  <TabItem value="python_objects">

[pickle](https://docs.python.org/3/library/pickle.html) 모듈로 직렬화할 수 있는 모든 것은 인테그레이션에 의해 생성된 아티팩트에 피클되고 추가됩니다. Dagster 내에서 해당 아티팩트를 읽을 때 내용이 피클 해제됩니다 (자세한 내용은 [Read artifacts](#read-wb-artifacts) 참조).

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

W&B는 여러 Pickle 기반 직렬화 모듈([pickle](https://docs.python.org/3/library/pickle.html), [dill](https://github.com/uqfoundation/dill), [cloudpickle](https://github.com/cloudpipe/cloudpickle), [joblib](https://github.com/joblib/joblib))을 지원합니다. 또한 [ONNX](https://onnx.ai/) 또는 [PMML](https://en.wikipedia.org/wiki/Predictive_Model_Markup_Language)과 같은 더 고급의 직렬화를 사용할 수 있습니다. 자세한 정보는 [Serialization](#serialization-configuration) 섹션을 참조하십시오.

  </TabItem>
  <TabItem value="wb_object">

어떠한 네이티브 W&B 객체(e.g [Table](../../ref/python/data-types/table.md), [Image](../../ref/python/data-types/image.md), [Graph](../../ref/python/data-types/graph.md))도 인테그레이션에 의해 생성된 아티팩트에 추가됩니다. 다음은 테이블을 사용하는 예제입니다.

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

  </TabItem>
  <TabItem value="wb_artifact">

복잡한 유스 케이스의 경우, 자신만의 Artifact 객체를 빌드해야 할 수도 있습니다. 인테그레이션은 여전히 양측 통합에서 메타데이터 확장을 제공하는 유용한 추가 기능을 제공합니다.

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

  </TabItem>
</Tabs>

### Configuration
`@op`, `@asset` 및 `@multi_asset`에 설정 사전인 `wandb_artifact_configuration`을 지정할 수 있습니다. 이 사전은 장식자 인수로 메타데이터에 전달되어야 합니다. 이는 W&B Artifacts의 IO Manager 읽기 및 쓰기를 제어하기 위해 필수적입니다.

`@op`의 경우 [Out](https://docs.dagster.io/_apidocs/ops#dagster.Out) 메타데이터 인수를 통해 출력 메타데이터에 위치합니다.
`@asset`의 경우 자산의 메타데이터 인수에 위치합니다.
`@multi_asset`의 경우 [AssetOut](https://docs.dagster.io/_apidocs/assets#dagster.AssetOut) 메타데이터 인수를 통해 각 출력 메타데이터에 위치합니다.

다음 코드 예제는 `@op`, `@asset` 및 `@multi_asset` 연산에서 사전을 설정하는 방법을 보여줍니다:

<Tabs
  defaultValue="op"
  values={[
    {label: 'Example for @op', value: 'op'},
    {label: 'Example for @asset', value: 'asset'},
    {label: 'Example for @multi_asset', value: 'multi_asset'},
  ]}>
  <TabItem value="op">

Example for `@op`:
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

  </TabItem>
  <TabItem value="asset">

Example for `@asset`:
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

`@asset`에는 이미 이름이 있으므로 구성에 이름을 전달할 필요가 없습니다. 인테그레이션은 아티팩트 이름을 자산 이름으로 설정합니다.

  </TabItem>
  <TabItem value="multi_asset">

Example for `@multi_asset`:

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

  </TabItem>
</Tabs>

켜진 속성으로 지원됩니다.
* `name`: (str) 이 아티팩트의 사람이 읽을 수 있는 이름입니다. 이는 UI에서 이 아티팩트를 식별하거나 use_artifact 호출에서 참조하는 방법입니다. 이름에는 문자, 숫자, 밑줄, 하이픈, 점이 포함될 수 있습니다. 프로젝트 내에서 고유해야 합니다. `@op`에 필수.
* `type`: (str) 아티팩트의 유형으로, 아티팩트를 조직하고 구분하는 데 사용됩니다. 일반적인 유형으로는 dataset이나 model이 있으며, 문자, 숫자, 밑줄, 하이픈, 점이 포함된 아무 문자열이나 사용할 수 있습니다. 출력이 이미 아티팩트가 아닌 경우에 필요합니다.
* `description`: (str) 아티팩트에 대한 설명을 제공합니다. 설명은 UI에서 마크다운으로 렌더링되며, 테이블, 링크 등을 작성하는 데 적합합니다.
* `aliases`: (list[str]) 아티팩트에 적용하려는 하나 이상의 별칭을 포함하는 배열입니다. 인테그레이션은 설정 여부에 관계없이 "latest" 태그를 이 목록에 추가합니다. 이는 모델 및 데이터셋의 버전 관리를 효과적으로 할 수 있는 방법입니다.
* [`add_dirs`](../../ref/python/artifact.md#add_dir): (list[dict[str, Any]]): 아티팩트에 포함할 각 로컬 디렉토리에 대한 설정을 포함하는 배열입니다. SDK의 동명 메소드와 같은 인수를 지원합니다.
* [`add_files`](../../ref/python/artifact.md#add_file): (list[dict[str, Any]]): 아티팩트에 포함할 각 로컬 파일에 대한 설정을 포함하는 배열입니다. SDK의 동명 메소드와 같은 인수를 지원합니다.
* [`add_references`](../../ref/python/artifact.md#add_reference): (list[dict[str, Any]]): 아티팩트에 포함할 각 외부 참조에 대한 설정을 포함하는 배열입니다. SDK의 동명 메소드와 같은 인수를 지원합니다.
* `serialization_module`: (dict) 사용될 직렬화 모듈의 설정입니다. 자세한 정보는 직렬화 섹션을 참조하십시오.
    * `name`: (str) 직렬화 모듈의 이름입니다. 허용된 값: `pickle`, `dill`, `cloudpickle`, `joblib`. 모듈은 로컬에서 사용 가능해야 합니다.
    * `parameters`: (dict[str, Any]) 직렬화 함수에 전달되는 선택적 인수입니다. 해당 모듈의 덤프 메소드와 동일한 파라미터를 허용합니다. e.g. `{"compress": 3, "protocol": 4}`.

고급 예제:
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

자산은 인테그레이션의 양측에 유용한 메타데이터로 실현됩니다:
* W&B 측: 소스 인테그레이션 이름과 버전, 사용된 파이썬 버전, 사용된 피클 프로토콜 버전 등.
* Dagster 측:
    * Dagster Run ID
    * W&B Run: ID, 이름, 경로, URL
    * W&B Artifact: ID, 이름, 유형, 버전, 크기, URL
    * W&B Entity
    * W&B Project

다음 이미지는 Dagster 자산에 추가된 W&B 메타데이터를 보여줍니다. 이 정보는 인테그레이션 없이는 제공되지 않을 것입니다.

![](/images/integrations/dagster_wb_metadata.png)

다음 이미지는 제공된 설정이 W&B Artifact에 유용한 메타데이터로 어떻게 확장되었는지 보여줍니다. 이 정보는 재현성 있고 유지 관리를 돕기 위한 것입니다. 인테그레이션 없이는 제공되지 않을 것입니다.

![](/images/integrations/dagster_inte_1.png)
![](/images/integrations/dagster_inte_2.png)
![](/images/integrations/dagster_inte_3.png)

:::info
mypy와 같은 정적 타입 체커를 사용하는 경우, 다음을 사용하여 설정 타입 정의 객체를 가져오세요:

```python
from dagster_wandb import WandbArtifactConfiguration
```
:::

### Using partitions

인테그레이션은 원래 [Dagster partitions](https://docs.dagster.io/concepts/partitions-schedules-sensors/partitions)를 지원합니다.

다음은 `DailyPartitionsDefinition`을 사용한 파티션 예제입니다.
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
이 코드는 각 파티션에 대해 하나의 W&B Artifact를 생성합니다. 이는 Asset name 아래의 Artifact 패널(UI)에서 `my_daily_partitioned_asset.2023-01-01`, `my_daily_partitioned_asset.2023-01-02`, `my_daily_partitioned_asset.2023-01-03` 등으로 찾을 수 있습니다. 여러 차원에 걸쳐 파티션된 자산은 각 차원이 점으로 나뉩니다. 예를 들어 `my_asset.car.blue`.

:::caution
인테그레이션은 한 run 내에서 여러 파티션의 실현을 허용하지 않습니다. 자산을 실현하려면 여러 run을 수행해야 합니다. 자산을 실현할 때 Dagit에서 실행할 수 있습니다.

![](/images/integrations/dagster_multiple_runs.png)
:::

#### 고급 사용법
- [파트리션된 작업](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/ops/partitioned_job.py)
- [간단한 파트리션된 자산](https://github.com/wandb/dagster/blob/master/examples/with_wandb/with_wandb/assets/simple_partitions_example.py)
- [다중 파트리션된 자산](https://github.com/wandb/dagster/blob/master/examples/with_wandb/with_wandb/assets/multi_partitions_example.py)
- [고급 파트리션 사용 예제](https://github.com/wandb/dagster/blob/master/examples/with_wandb/with_wandb/assets/advanced_partitions_example.py)

## Read W&B Artifacts
W&B Artifacts 읽기는 그것들을 작성하는 것과 유사합니다. `wandb_artifact_configuration`이라는 설정 사전이 `@op` 또는 `@asset`에 설정될 수 있습니다. 유일한 차이점은 출력 대신 입력에 설정을 해야 한다는 것입니다.

`@op`의 경우, 입력 메타데이터 내에서 [In](https://docs.dagster.io/_apidocs/ops#dagster.In) 메타데이터 인수를 통해 설정합니다. 아티팩트 이름을 명시적으로 지정해야 합니다.

`@asset`의 경우, 입력 메타데이터 내에서 [Asset](https://docs.dagster.io/_apidocs/assets#dagster.AssetIn) In 메타데이터 인수를 통해 설정합니다. 상위 자산의 이름이 일치해야 하므로 아티팩트 이름을 제공하지 않아야 합니다.

Integration 외부에서 생성된 아티팩트에 대한 의존성을 가지려면 [SourceAsset](https://docs.dagster.io/_apidocs/assets#dagster.SourceAsset)을 사용해야 합니다. 이는 항상 그 자산의 최신 버전을 읽습니다.

다음 예시들은 다양한 ops에서 아티팩트를 읽는 방법을 보여줍니다.

<Tabs
  defaultValue="op"
  values={[
    {label: 'From an @op', value: 'op'},
    {label: 'Created by another @asset', value: 'asset'},
    {label: 'Artifact created outside Dagster', value: 'outside_dagster'},
  ]}>
  <TabItem value="op">

`@op`에서 아티팩트를 읽기
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

  </TabItem>
  <TabItem value="asset">

다른 `@asset`에 의해 생성된 아티팩트를 읽기
```python
@asset(
   name="my_asset",
   ins={
       "artifact": AssetIn(
           # 입력 인수의 이름을 바꾸고 싶지 않다면 'key'를 제거할 수 있습니다.
           key="parent_dagster_asset_name",
           input_manager_key="wandb_artifacts_manager",
       )
   },
)
def read_artifact(context, artifact):
   context.log.info(artifact)
```

  </TabItem>
  <TabItem value="outside_dagster">

Dagster 외부에서 생성된 아티팩트를 읽기:

```python
my_artifact = SourceAsset(
   key=AssetKey("my_artifact"),  # W&B Artifacts의 이름입니다.
   description="Artifact created outside Dagster",
   io_manager_key="wandb_artifacts_manager",
)


@asset
def read_artifact(context, my_artifact):
   context.log.info(my_artifact)
```

  </TabItem>
</Tabs>

### Configuration
다음 설정은 IO 매니저가 장식된 함수에 제공해야 할 입력을 수집하고 제공하기 위해 사용됩니다. 다음 읽기 패턴이 지원됩니다.

1. 아티팩트 내의 지정된 객체를 얻으려면 get을 사용합니다:

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

2. 아티팩트 내에 포함된 파일의 로컬 경로를 얻으려면 get_path를 사용합니다:

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

3. 전체 아티팩트 객체를 얻으려면 (내용이 로컬로 다운로드된):
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
* `get`: (str) 아티팩트 상대 이름에 위치한 W&B 객체를 가져옵니다.
* `get_path`: (str) 아티팩트 상대 이름에 위치한 파일의 경로를 가져옵니다.

### Serialization configuration
기본적으로 인테그레이션은 표준 [pickle](https://docs.python.org/3/library/pickle.html) 모듈을 사용하지만 일부 객체는 호환되지 않습니다. 예를 들어, yield가 있는 함수는 피클로 저장하려 할 때 오류가 발생합니다.

우리는 더 많은 Pickle 기반 직렬화 모듈([dill](https://github.com/uqfoundation/dill), [cloudpickle](https://github.com/cloudpipe/cloudpickle), [joblib](https://github.com/joblib/joblib))을 지원합니다. 또한 직렬화된 문자열을 반환하거나 직접 아티팩트를 생성하여 [ONNX](https://onnx.ai/) 또는 [PMML](https://en.wikipedia.org/wiki/Predictive_Model_Markup_Language)와 같은 고급 직렬화를 사용하여 사용할 수 있습니다. 올바른 선택은 유스 케이스에 따라 다르며, 이 주제에 대한 사용 가능한 문헌을 참조하십시오.

### Pickle 기반 직렬화 모듈

:::caution
피클링은 알려진 취약점입니다. 보안 문제가 우려된다면 W&B 객체만 사용하십시오. 데이타에 서명을 하고, 해시 키를 자체 시스템에 저장하는 것을 권장합니다. 복잡한 사용자 사례에는 저희에게 연락하여 조언을 받으세요. 도와드리겠습니다.
:::

`wandb_artifact_configuration`의 `serialization_module` 사전을 통해 사용된 직렬화를 설정할 수 있습니다. Dagster를 실행하는 머신에서 모듈이 사용 가능할 수 있도록 하십시오.

인테그레이션은 아티팩트를 읽을 때 자동으로 어떤 직렬화 모듈을 사용할지 알게 됩니다.

현재 지원되는 모듈은 pickle, dill, cloudpickle 및 joblib입니다.

다음은 joblib으로 직렬화된 "model"을 생성하고이를 추론에 사용하는 단순화된 예제입니다.

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
    # 이건 실제 ML 모델은 아닙니다, 그러나 이것은 피클 모듈과 함께 불가능하기도 합니다.
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

### 고급 직렬화 형식 (ONNX, PMML)
ONNX 및 PMML과 같은 교환 파일 형식이 자주 사용됩니다. 인테그레이션은 해당 형식을 지원하지만 Pickle 기반 직렬화보다 약간의 작업이 필요합니다.

이러한 형식을 사용하기 위한 두 가지 방법이 있습니다.
1. 모델을 선택한 형식으로 변환한 다음, 해당 형식의 문자열 표현을 일반 Python 객체로 반환합니다. 인테그레이션은 해당 문자열을 피클로 저장합니다. 그런 다음 해당 문자열을 사용하여 모델을 다시 빌드할 수 있습니다.
2. 직렬화된 모델로 새 로컬 파일을 생성한 다음, `add_file` 설정을 사용하여 커스텀 아티팩트를 생성합니다.

다음은 ONNX를 사용하여 직렬화된 Scikit-learn 모델의 예입니다.

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
    # https://onnx.ai/sklearn-onnx/에서 영감을 받은 것입니다.

    # 모델을 훈련합니다.
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clr = RandomForestClassifier()
    clr.fit(X_train, y_train)

    # ONNX 형식으로 변환합니다
    initial_type = [("float_input", FloatTensorType([None, 4]))]
    onx = convert_sklearn(clr, initial_types=initial_type)

    # 아티팩트 작성 (모델 및 test_set)
    return onx.SerializeToString(), {"X_test": X_test, "y_test": y_test}

```python
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
    # https://onnx.ai/sklearn-onnx/에서 영감을 받은 것입니다.

    # ONNX Runtime으로 예측을 계산합니다.
    sess = rt.InferenceSession(my_onnx_model)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    pred_onx = sess.run(
        [label_name], {input_name: my_test_set["X_test"].astype(numpy.float32)}
    )[0]
    context.log.info(pred_onx)
    return pred_onx
```

### Using partitions

인테그레이션은 원래 [Dagster partitions](https://docs.dagster.io/concepts/partitions-schedules-sensors/partitions)를 지원합니다.

특정 또는 모든 파티션을 선택적으로 읽을 수 있습니다.

모든 파티션은 딕셔너리로 제공되며, 키와 값은 각각 파티션 키와 아티팩트 콘텐츠를 나타냅니다.

<Tabs
  defaultValue="all"
  values={[
    {label: 'Read all partitions', value: 'all'},
    {label: 'Read specific partitions', value: 'specific'},
  ]}>
  <TabItem value="all">

업스트림 `@asset`의 모든 파티션을 읽으며, 딕셔너리로 제공됩니다. 이 딕셔너리에서 키와 값은 각각 파티션 키와 아티팩트 콘텐츠와 관련이 있습니다.
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
  </TabItem>
  <TabItem value="specific">

`AssetIn`의 `partition_mapping` 설정을 통해 특정 파티션을 선택할 수 있습니다. 여기서는 `TimeWindowPartitionMapping`을 사용하고 있습니다.
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
  </TabItem>
</Tabs>

설정 객체인 `metadata`는 프로젝트에서 Weights & Biases (wandb)가 다른 아티팩트 파티션과 상호 작용하는 방법을 구성하는 데 사용됩니다.

`metadata` 객체에는 `wandb_artifact_configuration`이라는 이름의 키가 포함되어 있으며, 여기에는 `partitions`라는 중첩 객체가 포함되어 있습니다.

`partitions` 객체는 각 파티션의 이름을 해당 설정에 매핑합니다. 각 파티션의 구성은 어떻게 데이터를 가져와야 하는지를 지정할 수 있으며, 이 설정에는 각 파티션의 요구 사항에 따라 각각 `get`, `version`, `alias` 키가 포함될 수 있습니다.

**구성 키**

1. `get`:
`get` 키는 데이터를 가져올 W&B 객체(Table, Image 등)의 이름을 지정합니다.
2. `version`:
`version` 키는 특정 아티팩트 버전을 가져오고자 할 때 사용됩니다.
3. `alias`:
`alias` 키는 아티팩트를 별칭으로 가져옵니다.

**와일드 카드 구성**

와일드 카드 `"*"`은 구성되지 않은 모든 파티션을 의미합니다. 이는 `partitions` 객체에 명시적으로 언급되지 않은 파티션에 대한 기본 구성을 제공합니다.

예를 들어,

```python
"*": {
    "get": "default_table_name",
},
```
이 구성은 명시적으로 구성되지 않은 모든 파티션에 대해 `default_table_name`이라는 테이블에서 데이터를 가져온다는 의미입니다.

**특정 파티션 구성**

와일드 카드 구성을 특정 파티션에 대한 별도의 구성을 제공하여 덮어쓸 수 있습니다.

예를 들어,

```python
"yellow": {
    "get": "custom_table_name",
},
```

이 구성은 `yellow`라는 파티션에 대해 `custom_table_name`이라는 테이블에서 데이터를 가져와 와일드 카드 구성을 덮어쓴다는 의미입니다.

**버전 관리 및 별칭**

버전 관리와 별칭 설정 목적을 위해, 구성에서 특정 `version` 및 `alias` 키를 제공할 수 있습니다.

버전의 경우,

```python
"orange": {
    "version": "v0",
},
```

이 구성은 `orange` 아티팩트 파티션의 `v0` 버전에서 데이터를 가져오겠다는 것을 의미합니다.

별칭의 경우,

```python
"blue": {
    "alias": "special_alias",
},
```

이 구성은 `blue`로 구성된 `special_alias`라는 별칭을 가진 아티팩트 파티션의 `default_table_name` 테이블에서 데이터를 가져온다는 의미입니다.

### 고급 사용법
인테그레이션의 고급 사용법에 대한 자세한 코드 예시는 다음을 참조하세요:
* [자산 고급 사용 예제](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/assets/advanced_example.py)
* [파트리션된 작업 예제](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/ops/partitioned_job.py)
* [모델을 모델 레지스트리에 연결하는 예제](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/assets/model_registry_example.py)

## Using W&B Launch

:::caution
활발히 개발 중인 베타 제품
Launch에 관심이 있는 고객은 W&B Launch의 고객 파일럿 프로그램에 가입하시려면 계정 팀에 연락하십시오.
파일럿 고객은 AWS EKS 또는 SageMaker를 사용해야 베타 프로그램 자격요건을 충족할 수 있습니다. 우리는 궁극적으로 추가적인 플랫폼을 지원할 계획입니다.
:::

계속 진행하기 전에 W&B Launch를 사용하는 방법에 대한 충분한 이해가 필요합니다. Launch 가이드 읽기를 고려하세요: /guides/launch.

Dagster 인테그레이션은 다음을 돕습니다:
* Dagster 인스턴스 내에서 하나 이상의 Launch 에이전트를 실행합니다.
* Dagster 인스턴스 내에서 로컬 Launch 작업을 수행합니다.
* 온프레미스 또는 클라우드에서 원격 Launch 작업을 수행합니다.

### Launch agents
인테그레이션은 `run_launch_agent`라는 import 가능한 `@op`을 제공합니다. 이는 Launch Agent를 시작하고 수동으로 중지될 때까지 장기 실행 프로세스로 실행됩니다.

Agents는 launch 큐를 폴링하고 작업을 실행하거나 외부 서비스로 전달하여 실행하는 프로세스입니다.

구성에 대한 참조 문서를 확인하세요

Launchpad에서 모든 속성에 대한 유용한 설명도 볼 수 있습니다.

![](/images/integrations/dagster_launch_agents.png)

간단한 예제
```python
# config.yaml에 추가하세요
# 대안으로는 Dagit's Launchpad 또는 JobDefinition.execute_in_process에서 설정할 수 있습니다
# 참조: https://docs.dagster.io/concepts/configuration/config-schema#specifying-runtime-configuration
resources:
 wandb_config:
   config:
     entity: my_entity # 이 부분을 자신의 W&B 엔터티로 교체하세요
     project: my_project # 이 부분을 자신의 W&B 프로젝트로 교체하세요
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
인테그레이션은 `run_launch_job`이라는 import 가능한 `@op`을 제공합니다. 이는 Launch 작업을 실행합니다.

Launch 작업은 실행을 위해 큐에 할당됩니다. 큐를 생성하거나 기본 큐를 사용할 수 있습니다. 해당 큐를 수신 중인 활성 에이전트가 있는지 확인하세요. Dagster 인스턴스 내에서 에이전트를 실행할 수 있지만, Kubernetes에서 배포 가능한 에이전트를 사용하는 것을 고려할 수도 있습니다.

구성에 대한 참조 문서를 확인하세요.

Launchpad에서 모든 속성에 대한 유용한 설명도 볼 수 있습니다.

![](/images/integrations/dagster_launch_jobs.png)

간단한 예제
```python
# config.yaml에 추가하세요
# 대안으로는 Dagit's Launchpad 또는 JobDefinition.execute_in_process에서 설정할 수 있습니다
# 참조: https://docs.dagster.io/concepts/configuration/config-schema#specifying-runtime-configuration
resources:
 wandb_config:
   config:
     entity: my_entity # 이 부분을 자신의 W&B 엔터티로 교체하세요
     project: my_project # 이 부분을 자신의 W&B 프로젝트로 교체하세요
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

from dagster import job,
  make_values_resource

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
   run_launch_job.alias("my_launched_job")() # 우리는 에일리어스로 job 이름을 변경합니다.
```

## Best practices

1. IO 매니저를 사용하여 Artifacts를 읽고 쓰십시오.
`Artifact.download()`나 `Run.log_artifact()`를 직접 사용할 필요가 없습니다. 이러한 메소드는 인테그레이션에서 처리됩니다. 저장할 데이터를 아티팩트에 반환하기 만 하면 되고 나머지는 인테그레이션이 처리합니다. 이는 W&B에서 아티팩트의 계보를 더 좋게 만들어줍니다.

2. 복잡한 유스 케이스에만 직접 아티팩트 객체를 생성하십시오.
Python 객체와 W&B 객체는 ops/assets에서 반환되어야 합니다. 인테그레이션은 아티팩트를 묶는 과정을 처리합니다.
복잡한 유스 케이스의 경우, Dagster 작업에서 직접 아티팩트를 생성할 수 있습니다. 인테그레이션에 아티팩트 객체를 전달하여 소스 인테그레이션 이름 및 버전, 사용된 파이썬 버전, 피클 프로토콜 버전 등의 메타데이터를 확장하는 것을 추천합니다.

3. 메타데이터를 통해 Artifacts에 파일, 디렉토리 및 외부 참조를 추가하십시오.
통합 `wandb_artifact_configuration` 객체를 사용하여 파일, 디렉토리 또는 외부 참조(Amazon S3, GCS, HTTP...)를 추가하십시오. 자세한 정보는 [Artifacts 설정 섹션](#configuration-1)의 고급 예제를 참조하십시오.

4. 아티팩트를 생성할 때 @asset을 사용하십시오.
아티팩트는 자산입니다. Dagster가 해당 자산을 관리할 때는 asset을 사용하는 것이 권장됩니다. 이는 Dagit 자산 카탈로그에서 더 나은 관찰성을 제공합니다.

5. Dagster 외부에서 생성된 Artifact를 소비하기 위해 SourceAsset을 사용하십시오.
이는 인테그레이션을 활용하여 외부 생성 아티팩트를 읽을 수 있습니다. 그렇지 않으면 인테그레이션에서 생성된 아티팩트만 사용할 수 있습니다.

6. 대형 모델을 위한 전용 컴퓨팅에서의 트레이닝 오케스트레이션을 위해 W&B Launch를 사용하십시오.
작은 모델은 Dagster 클러스터 내에서 훈련할 수 있으며, GPU 노드가 있는 Kubernetes 클러스터에서 Dagster를 실행할 수 있습니다. 우리는 대형 모델 트레이닝을 위해 W&B Launch 사용을 권장합니다. 이는 인스턴스의 과부하를 방지하고 더 적절한 컴퓨팅을 제공할 것입니다.

7. Dagster 내에서 실험을 추적할 때, W&B Run ID를 Dagster Run ID로 설정하십시오.
W&B Run ID를 Dagster Run ID 또는 사용자의 선택한 문자열로 설정하고, [Run 재개 가능 설정](../runs/resuming.md)을 통해 실험이 재개 가능하도록 하십시오. 이 추천을 따르면, Dagster 내에서 모델을 훈련할 때 W&B 메트릭 및 W&B Artifacts가 동일한 W&B Run에 저장되도록 합니다.

W&B Run ID를 Dagster Run ID로 설정하는 방법:
```python
wandb.init(
    id=context.run_id,
    resume="allow",
    ...
)
```

또는 자신만의 W&B Run ID를 선택하고 IO 매니저 설정에 전달하십시오:
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

8. 큰 W&B Artifacts의 경우 필요한 데이터만 get 또는 get_path를 사용하여 수집하십시오.
기본적으로 인테그레이션은 전체 아티팩트를 다운로드합니다. 매우 큰 아티팩트를 사용하는 경우에는 필요한 특정 파일이나 객체만 수집할 수 있습니다. 이는 속도 및 자원 활용을 향상시킬 것입니다.

9. Python 객체의 경우 피클링 모듈을 유스 케이스에 맞게 조정하십시오.
기본적으로 W&B 통합은 표준 [pickle](https://docs.python.org/3/library/pickle.html) 모듈을 사용합니다. 그러나 일부 객체는 사용이 불가능할 수 있습니다. 예를 들어, yield가 있는 함수는 피클로 저장하려 할 때 오류를 발생합니다. W&B는 다른 Pickle 기반 직렬화 모듈([dill](https://github.com/uqfoundation/dill), [cloudpickle](https://github.com/cloudpipe/cloudpickle), [joblib](https://github.com/joblib/joblib))을 지원합니다.

더 고급 직렬화 옵션으로 [ONNX](https://onnx.ai/)나 [PMML](https://en.wikipedia.org/wiki/Predictive_Model_Markup_Language)를 사용하는 것도 가능하며, 직렬화된 문자열을 반환하거나 직접 아티팩트를 생성할 수 있습니다. 올바른 선택은 유스 케이스에 달려 있습니다. 이 주제에 대한 사용 가능한 문헌을 참조하십시오.
```
---
description: Guide on how to integrate W&B with Dagster.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Dagster

Dagster와 W&B (W&B)를 사용하여 MLOps 파이프라인을 조율하고 ML 자산을 관리하세요. Dagster와의 통합을 통해 다음 작업을 쉽게 수행할 수 있습니다:

* [W&B Artifacts](../artifacts/intro.md) 사용 및 생성.
* [W&B 모델 레지스트리](../model_registry/intro.md)에서 등록된 모델 사용 및 생성.
* [W&B Launch](../launch/intro.md)를 사용하여 전용 컴퓨트에서 학습 작업 실행.
* ops 및 자산에서 [wandb](../../ref/python/README.md) 클라이언트 사용.

W&B Dagster 통합은 W&B 특정 Dagster 리소스 및 IO 매니저를 제공합니다:

* `wandb_resource`: W&B API에 인증하고 통신하는 데 사용되는 Dagster 리소스입니다.
* `wandb_artifacts_io_manager`: W&B Artifacts를 사용하는 Dagster IO 매니저입니다.

다음 가이드는 Dagster에서 W&B를 사용하기 위한 전제 조건을 충족하는 방법, ops 및 자산에서 W&B Artifacts를 생성하고 사용하는 방법, W&B Launch를 사용하는 방법 및 권장 모범 사례를 보여줍니다.

## 시작하기 전에
Weights and Biases 내에서 Dagster를 사용하려면 다음 리소스가 필요합니다:
1. **W&B API 키**.
2. **W&B 엔티티 (사용자 또는 팀)**: 엔티티는 W&B 실행과 아티팩트를 전송하는 사용자 이름이나 팀 이름입니다. 실행을 로그하기 전에 W&B 앱 UI에서 계정이나 팀 엔티티를 만드세요. 엔티티를 지정하지 않으면, 실행은 보통 사용자 이름인 기본 엔티티로 전송됩니다. **프로젝트 기본값** 아래 설정에서 기본 엔티티를 변경할 수 있습니다.
3. **W&B 프로젝트**: [W&B 실행](../runs/intro.md)이 저장되는 프로젝트 이름입니다.

W&B 앱에서 해당 사용자 또는 팀의 프로필 페이지를 확인하여 W&B 엔티티를 찾으세요. 기존 W&B 프로젝트를 사용하거나 새로운 프로젝트를 만들 수 있습니다. 새 프로젝트는 W&B 앱 홈페이지 또는 사용자/팀 프로필 페이지에서 생성할 수 있습니다. 프로젝트가 존재하지 않으면 처음 사용할 때 자동으로 생성됩니다. 다음 지침은 API 키를 얻는 방법을 보여줍니다:

### API 키 얻는 방법
1. [W&B에 로그인](https://wandb.ai/login)하세요. 참고: W&B 서버를 사용하는 경우 관리자에게 인스턴스 호스트 이름을 요청하세요.
2. [인증 페이지](https://wandb.ai/authorize)로 이동하거나 사용자/팀 설정에서 API 키를 수집하세요. 프로덕션 환경에서는 해당 키를 소유하는 [서비스 계정](https://docs.wandb.ai/guides/technical-faq/general#what-is-a-service-account-and-why-is-it-useful)을 사용하는 것이 좋습니다.
3. 해당 API 키에 대한 환경 변수를 설정하세요 export `WANDB_API_KEY=YOUR_KEY`.


다음 예제는 Dagster 코드에서 API 키를 지정하는 위치를 보여줍니다. `wandb_config` 중첩 딕셔너리 내에서 엔티티와 프로젝트 이름을 지정하세요. 다른 W&B 프로젝트를 사용하려면 다른 ops/자산에 다른 `wandb_config` 값을 전달할 수 있습니다. 전달할 수 있는 키에 대한 자세한 정보는 아래 구성 섹션을 참조하세요.


<Tabs
  defaultValue="job"
  values={[
    {label: '작업용 구성', value: 'job'},
    {label: '자산을 사용하는 리포지토리용 구성', value: 'repository'},
  ]}>
  <TabItem value="job">

예시: `@job`용 구성
```python
# config.yaml에 이를 추가하세요
# 대안으로 Dagit의 Launchpad 또는 JobDefinition.execute_in_process에서 구성을 설정할 수 있습니다
# 참조: https://docs.dagster.io/concepts/configuration/config-schema#specifying-runtime-configuration
resources:
 wandb_config:
   config:
     entity: my_entity # 여기에 W&B 엔티티를 대체하세요
     project: my_project # 여기에 W&B 프로젝트를 대체하세요


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


예시: 자산을 사용하는 `@repository`용 구성

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
                       "entity": "my_entity", # 여기에 W&B 엔티티를 대체하세요
                       "project": "my_project", # 여기에 W&B 프로젝트를 대체하세요
                   }
               }
           },
       ),
   ]
```
이 예시에서는 `@job`용 예시와 달리 IO 매니저 캐시 지속 시간을 구성하고 있습니다.

  </TabItem>
</Tabs>

### 구성
다음 구성 옵션은 통합에 의해 제공되는 W&B 특정 Dagster 리소스 및 IO 매니저에 사용되는 설정입니다.

* `wandb_resource`: W&B API와 통신하는 데 사용되는 Dagster [리소스](https://docs.dagster.io/concepts/resources)입니다. 제공된 API 키를 사용하여 자동으로 인증합니다. 속성:
    * `api_key`: (str, 필수): W&B API와 통신하는 데 필요한 W&B API 키입니다.
    * `host`: (str, 선택 사항): 사용하고자 하는 API 호스트 서버입니다. W&B 서버를 사용하는 경우에만 필요합니다. 기본값은 공용 클라우드 호스트입니다: [https://api.wandb.ai](https://api.wandb.ai)
* `wandb_artifacts_io_manager`: W&B Artifacts를 사용하는 Dagster [IO 매니저](https://docs.dagster.io/concepts/io-management/io-managers)입니다. 속성:
    * `base_dir`: (int, 선택 사항) 로컬 저장소 및 캐싱에 사용되는 기본 디렉터리입니다. W&B Artifacts 및 W&B 실행 로그는 해당 디렉터리에서 작성 및 읽기가 이루어집니다. 기본적으로 `DAGSTER_HOME` 디렉터리를 사용합니다.
    * `cache_duration_in_minutes`: (int, 선택 사항) W&B Artifacts 및 W&B 실행 로그를 로컬 저장소에 보관해야 하는 시간을 정의합니다. 지정된 시간 동안 열리지 않은 파일 및 디렉터리만 캐시에서 제거됩니다. 캐시 정리는 IO 매니저 실행이 끝날 때 발생합니다. 캐싱을 완전히 비활성화하려면 0으로 설정할 수 있습니다. 캐싱은 동일한 기계에서 실행되는 작업 간에 아티팩트가 재사용될 때 속도를 향상시킵니다. 기본값은 30일입니다.
    * `run_id`: (str, 선택 사항): 이 실행에 대한 고유 ID로, 이어서 사용하는 데 필요합니다. 프로젝트 내에서 고유해야 하며, 실행을 삭제하면 ID를 재사용할 수 없습니다. 짧은 설명 이름에는 이름 필드를 사용하거나 실행을 비교하기 위해 하이퍼파라미터를 저장하는 데 config를 사용하세요. ID에는 다음과 같은 특수 문자를 포함할 수 없습니다: `/\#?%:..` Dagster 내부에서 실험 추적을 수행할 때 실행을 이어서 사용하기 위해 실행 ID를 설정해야 합니다. 기본적으로 Dagster 실행 ID로 설정됩니다. 예: `7e4df022-1bf2-44b5-a383-bb852df4077e`.
    * `run_name`: (str, 선택 사항) 이 실행에 대한 짧은 표시 이름으로, UI에서 이 실행을 식별하는 데 사용됩니다. 기본적으로 다음 형식의 문자열로 설정됩니다. dagster-run-[Dagster 실행 ID의 처음 8자] 예: `dagster-run-7e4df022`.
    * `run_tags`: (list[str], 선택 사항): UI에서 이 실행의 태그 목록을 채우는 문자열 목록입니다. 태그는 실행을 함께 구성하거나 "기준" 또는 "프로덕션"과 같은 임시 레이블을 적용하는 데 유용합니다. UI에서 태그를 추가하거나 제거하거나 특정 태그가 있는 실행만 필터링하는 것이 쉽습니다. 통합에 의해 사용되는 모든 W&B 실행에는 `dagster_wandb` 태그가 있습니다.

## W&B Artifacts 사용하기

W&B Artifact와의 통합은 Dagster IO 매니저에 의존합니다.

[IO 매니저](https://docs.dagster.io/concepts/io-management/io-managers)는 자산 또는 op의 출력을 저장하고 하위 자산 또는 op에 입력으로 로드하는 책임이 있는 사용자 제공 객체입니다. 예를 들어, IO 매니저는 파일시스템의 파일에서 객체를 저장하고 로드할 수 있습니다.

통합은 W&B Artifacts에 대한 IO 매니저를 제공합니다. 이를 통해 모든 Dagster `@op` 또는 `@asset`이 기본적으로 W&B Artifacts를 생성하고 사용할 수 있습니다. 다음은 Python 리스트를 포함하는 dataset 유형의 W&B Artifact를 생성하는 `@asset`의 간단한 예입니다.

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
    return [1, 2, 3] # 이것은 Artifact에 저장됩니다
```

Artifacts를 작성하기 위해 메타데이터 구성으로 `@op`, `@asset` 및 `@multi_asset`을 주석 처리할 수 있습니다. 마찬가지로 Dagster 외부에서 생성된 W&B Artifacts도 사용할 수 있습니다.

## W&B Artifacts 작성하기
계속하기 전에 W&B Artifacts 사용 방법을 잘 이해하는 것이 좋습니다. [Artifacts 가이드](../artifacts/intro.md)를 읽어보세요.

Python 함수에서 객체를 반환하여 W&B Artifact를 작성합니다. 다음 객체는 W&B에서 지원됩니다:
* Python 객체 (int, dict, list...)
* W&B 객체 (Table, Image, Graph...)
* W&B Artifact 객체

다음 예제는 Dagster 자산(`@asset`)으로 W&B Artifacts를 작성하는 방법을 보여줍니다:


<Tabs
  defaultValue="python_objects"
  values={[
    {label: 'Python 객체', value: 'python_objects'},
    {label: 'W&B 객체', value: 'wb_object'},
    {label: 'W&B Artifacts', value: 'wb_artifact'},
  ]}>
  <TabItem value="python_objects">

[pickle](https://docs.python.org/3/library/pickle.html) 모듈로 직렬화될 수 있는 모든 것은 통합에 의해 생성된 Artifact에 피클링되어 추가됩니다. 내용은 Dagster 내부에서 해당 Artifact를 읽을 때 unpickling됩니다 (자세한 내용은 [아티팩트 읽기](#read-wb-artifacts) 참조).

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


W&B는 여러 Pickle 기반 직렬화 모듈([pickle](https://docs.python.org/3/library/pickle.html), [dill](https://github.com/uqfoundation/dill), [cloudpickle](https://github.com/cloudpipe/cloudpickle), [joblib](https://github.com/joblib/joblib))을 지원합니다. [ONNX](https://onnx.ai/) 또는 [PMML](https://en.wikipedia.org/wiki/Predictive_Model_Markup_Language)과 같은 더 고급 직렬화를 사용할 수도 있습니다. 자세한 정보는 [직렬화](#serialization-configuration) 섹션을 참조하세요.

  </TabItem>
  <TabItem value="wb_object">

통합에 의해 생성된 Artifact에 추가되는 모든 네이티브 W&B 객체 (예: [Table](../../ref/python/data-types/table.md), [Image](../../ref/python/data-types/image.md), [Graph](../../ref/python/data-types/graph.md))입니다. 다음은 Table을 사용한 예입니다.

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

복잡한 사용 사례의 경우, 자체 Artifact 객체를 구축해야 할 수도 있습니다. 통합은 여전히 통합 양쪽에서 메타데이터를 보강하는 유용한 추가 기능을 제공합니다.

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

### 구성
`@op`, `@asset` 및 `@multi_asset`에서 wandb_artifact_configuration이라는 구성 사전을 설정할 수 있습니다. 이 사전은 메타데이터로 데코레이터 인수에 전달되어야 합니다. 이 구성은 W&B 아티팩트의 IO 관리자 읽기 및 쓰기를 제어하는 데 필요합니다.

`@op`의 경우, [Out](https://docs.dagster.io/_apidocs/ops#dagster.Out) 메타데이터 인수를 통해 출력 메타데이터에 위치합니다.
`@asset`의 경우, 자산의 메타데이터 인수에 위치합니다.
`@multi_asset`의 경우, [AssetOut](https://docs.dagster.io/_apidocs/assets#dagster.AssetOut) 메타데이터 인수를 통해 각 출력 메타데이터에 위치합니다.

다음 코드 예제는 `@op`, `@asset` 및 `@multi_asset` 계산에 사전을 구성하는 방법을 보여줍니다:

<Tabs
  defaultValue="op"
  values={[
    {label: '@op 예제', value: 'op'},
    {label: '@asset 예제', value: 'asset'},
    {label: '@multi_asset 예제', value: 'multi_asset'},
  ]}>
  <TabItem value="op">

`@op` 예제:
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

`@asset` 예제:
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

구성을 통해 이름을 전달할 필요가 없습니다. 왜냐하면 @asset에 이미 이름이 있기 때문입니다. 통합은 자산 이름으로 아티팩트 이름을 설정합니다.

  </TabItem>
  <TabItem value="multi_asset">

`@multi_asset` 예제:

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

지원되는 속성:
* `name`: (str) 이 아티팩트의 사람이 읽을 수 있는 이름으로, UI에서 이 아티팩트를 식별하거나 use_artifact 호출에서 참조하는 방법입니다. 이름에는 글자, 숫자, 밑줄, 하이픈 및 점이 포함될 수 있습니다. 이름은 프로젝트 전체에서 고유해야 합니다. `@op`에 필요합니다.
* `type`: (str) 아티팩트의 유형으로, 아티팩트를 구성하고 구별하는 데 사용됩니다. 일반적인 유형에는 데이터세트 또는 모델이 포함되지만 글자, 숫자, 밑줄, 하이픈 및 점을 포함하는 모든 문자열을 사용할 수 있습니다. 출력이 이미 아티팩트가 아닌 경우 필요합니다.
* `description`: (str) 아티팩트의 설명을 제공하는 자유 텍스트입니다. 설명은 UI에서 마크다운으로 렌더링되므로 표, 링크 등을 배치하기에 좋은 곳입니다.
* `aliases`: (list[str]) 하나 이상의 별칭을 적용하려는 별칭 배열입니다. 통합은 “latest” 태그를 해당 목록에 설정 여부와 관계없이 추가합니다. 이것은 모델 및 데이터세트의 버전 관리를 관리하는 효과적인 방법입니다.
* [`add_dirs`](../../ref/python/artifact.md#add_dir): (list[dict[str, Any]]): 아티팩트에 포함할 각 로컬 디렉터리에 대한 구성을 포함하는 배열입니다. SDK의 동명 메서드와 동일한 인수를 지원합니다.
* [`add_files`](../../ref/python/artifact.md#add_file): (list[dict[str, Any]]): 아티팩트에 포함할 각 로컬 파일에 대한 구성을 포함하는 배열입니다. SDK의 동명 메서드와 동일한 인수를 지원합니다.
* [`add_references`](../../ref/python/artifact.md#add_reference): (list[dict[str, Any]]): 아티팩트에 포함할 각 외부 참조에 대한 구성을 포함하는 배열입니다. SDK의 동명 메서드와 동일한 인수를 지원합니다.
* `serialization_module`: (dict) 사용할 직렬화 모듈의 구성입니다. 자세한 내용은 직렬화 섹션을 참조하십시오.
    * `name`: (str) 직렬화 모듈의 이름입니다. 허용되는 값: `pickle`, `dill`, `cloudpickle`, `joblib`. 모듈이 로컬에 사용 가능해야 합니다.
    * `parameters`: (dict[str, Any]) 직렬화 함수에 전달된 선택적 인수입니다. 해당 모듈의 dump 메서드와 동일한 파라미터를 수용합니다. 예: `{"compress": 3, "protocol": 4}`.

고급 예제:
```python
@asset(
   name="my_advanced_artifact",
   metadata={
       "wandb_artifact_configuration": {
           "type": "dataset",
           "description": "내 *마크다운* 설명",
           "aliases": ["my_first_alias", "my_second_alias"],
           "add_dirs": [
               {
                   "name": "내 디렉터리",
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
                   "name": "이미지에 대한 외부 HTTP 참조",
               },
               {
                   "uri": "s3://my-bucket/datasets/mnist",
                   "name": "외부 S3 참조",
               },
           ],
       }
   },
   io_manager_key="wandb_artifacts_manager",
)
def create_advanced_artifact():
   return [1, 2, 3]
```

자산은 통합의 양쪽에서 유용한 메타데이터와 함께 구체화됩니다:
* W&B 측면: 소스 통합 이름 및 버전, 사용된 파이썬 버전, 피클 프로토콜 버전 등.
* Dagster 측면:
    * Dagster 실행 ID
    * W&B 실행: ID, 이름, 경로, URL
    * W&B 아티팩트: ID, 이름, 유형, 버전, 크기, URL
    * W&B 엔티티
    * W&B 프로젝트

다음 이미지는 Dagster 자산에 추가된 W&B의 메타데이터를 보여줍니다. 이 정보는 통합 없이는 사용할 수 없습니다.

![](/images/integrations/dagster_wb_metadata.png)

다음 이미지는 제공된 구성이 W&B 아티팩트에 유용한 메타데이터로 풍부해진 방법을 보여줍니다. 이 정보는 재현성 및 관리에 도움이 될 것입니다. 이 정보는 통합 없이는 사용할 수 없습니다.

![](/images/integrations/dagster_inte_1.png)
![](/images/integrations/dagster_inte_2.png)
![](/images/integrations/dagster_inte_3.png)

:::info
mypy와 같은 정적 타입 검사기를 사용하는 경우, 구성 유형 정의 객체를 다음과 같이 가져오십시오:

```python
from dagster_wandb import WandbArtifactConfiguration
```
:::

### 파티션 사용

통합은 [Dagster 파티션](https://docs.dagster.io/concepts/partitions-schedules-sensors/partitions)을 기본적으로 지원합니다.

다음은 `DailyPartitionsDefinition`을 사용하여 파티션화된 예입니다.
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
    context.log.info(f"{partition_key}에 대한 파티션화된 자산 생성")
    return random.randint(0, 100)
```
이 코드는 각 파티션에 대해 하나의 W&B 아티팩트를 생성합니다. 그들은 자산 이름 아래의 아티팩트 패널(UI)에서 찾을 수 있으며, 파티션 키가 추가됩니다. 예를 들어 `my_daily_partitioned_asset.2023-01-01`, `my_daily_partitioned_asset.2023-01-02`, `my_daily_partitioned_asset.2023-01-03` 등입니다. 여러 차원에 걸쳐 파티션화된 자산은 각 차원이 점으로 구분됩니다. 예: `my_asset.car.blue`.

:::caution
통합은 한 번의 실행 내에서 여러 파티션의 구체화를 허용하지 않습니다. 자산을 구체화하려면 여러 번의 실행이 필요합니다. 이는 자산을 구체화할 때 Dagit에서 실행할 수 있습니다.

![](/images/integrations/dagster_multiple_runs.png)
:::

#### 고급 사용법
- [파티션화된 작업](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/ops/partitioned_job.py)
- [간단한 파티션화된 자산](https://github.com/wandb/dagster/blob/master/examples/with_wandb/with_wandb/assets/simple_partitions_example.py)
- [다중 파티션화된 자산](https://github.com/wandb/dagster/blob/master/examples/with_wandb/with_wandb/assets/multi_partitions_example.py)
- [고급 파티션화된 사용법](https://github.com/wandb/dagster/blob/master/examples/with_wandb/with_wandb/assets/advanced_partitions_example.py)

## W&B 아티팩트 읽기
W&B 아티팩트를 읽는 것은 그것들을 쓰는 것과 유사합니다. `@op` 또는 `@asset`에 `wandb_artifact_configuration`이라는 구성 사전을 설정할 수 있습니다. 유일한 차이점은 구성을 출력 대신 입력에 설정해야 한다는 것입니다.

`@op`의 경우, [In](https://docs.dagster.io/_apidocs/ops#dagster.In) 메타데이터 인수를 통해 입력 메타데이터에 위치합니다. 아티팩트의 이름을 명시적으로 전달해야 합니다.

`@asset`의 경우, [Asset](https://docs.dagster.io/_apidocs/assets#dagster.AssetIn) In 메타데이터 인수를 통해 입력 메타데이터에 위치합니다. 부모 자산의 이름과 일치해야 하므로 아티팩트 이름을 전달해서는 안 됩니다.

통합 외부에서 생성된 아티팩트에 대한 의존성이 있으면 [SourceAsset](https://docs.dagster.io/_apidocs/assets#dagster.SourceAsset)을 사용해야 합니다. 항상 해당 자산의 최신 버전을 읽습니다.

다양한 ops에서 아티팩트를 읽는 방법을 보여주는 다음 예제입니다.

<Tabs
  defaultValue="op"
  values={[
    {label: '@op에서', value: 'op'},
    {label: '다른 @asset에 의해 생성됨', value: 'asset'},
    {label: 'Dagster 외부에서 생성된 아티팩트', value: 'outside_dagster'},
  ]}>
  <TabItem value="op">

`@op`에서 아티팩트 읽기
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

다른 `@asset`에 의해 생성된 아티팩트 읽기
```python
@asset(
   name="my_asset",
   ins={
       "artifact": AssetIn(
           # 입력 인수의 이름을 바꾸고 싶지 않다면 'key'를 제거할 수 있습니다
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

Dagster 외부에서 생성된 아티팩트 읽기:

```python
my_artifact = SourceAsset(
   key=AssetKey("my_artifact"),  # W&B 아티팩트의 이름
   description="Dagster 외부에서 생성된 아티팩트",
   io_manager_key="wandb_artifacts_manager",
)


@asset
def read_artifact(context, my_artifact):
   context.log.info(my_artifact)
```

  </TabItem>
</Tabs>

### 구성
다음 구성은 IO 관리자가 수집하여 장식된 함수에 입력으로 제공해야 하는 것을 나타냅니다. 다음 읽기 패턴이 지원됩니다.

1. 아티팩트 내에 포함된 명명된 개체를 가져오려면 get을 사용하십시오:

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


2. 아티팩트 내에 포함된 다운로드된 파일의 로컬 경로를 가져오려면 get_path를 사용하십시오:

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

3. 내용이 로컬로 다운로드된 전체 아티팩트 개체를 가져오려면:
```python
@asset(
   ins={
       "artifact": AssetIn(
           key="my_artifact",
           input_manager_key="wandb_artifacts_manager",
       )
   },
)
def get_artifact(context

### 고급 직렬화 포맷 (ONNX, PMML)
ONNX 및 PMML과 같은 교환 파일 포맷을 사용하는 것이 일반적입니다. 이러한 포맷은 지원되지만, Pickle 기반 직렬화보다 조금 더 많은 작업이 필요합니다.

이러한 포맷을 사용하는 데에는 두 가지 다른 방법이 있습니다.
1. 모델을 선택한 포맷으로 변환한 후, 일반 Python 객체처럼 그 포맷의 문자열 표현을 반환합니다. 통합은 그 문자열을 피클합니다. 그런 다음 해당 문자열을 사용하여 모델을 다시 빌드할 수 있습니다.
2. 직렬화된 모델을 포함하는 새로운 로컬 파일을 생성한 후, add_file 구성을 사용하여 해당 파일로 사용자 지정 아티팩트를 빌드합니다.

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
    # https://onnx.ai/sklearn-onnx/ 에서 영감을 받았습니다.

    # 모델 학습
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clr = RandomForestClassifier()
    clr.fit(X_train, y_train)

    # ONNX 포맷으로 변환
    initial_type = [("float_input", FloatTensorType([None, 4]))]
    onx = convert_sklearn(clr, initial_types=initial_type)

    # 아티팩트 작성 (모델 + 테스트 세트)
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
    # https://onnx.ai/sklearn-onnx/ 에서 영감을 받았습니다.

    # ONNX Runtime으로 예측값 계산
    sess = rt.InferenceSession(my_onnx_model)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    pred_onx = sess.run(
        [label_name], {input_name: my_test_set["X_test"].astype(numpy.float32)}
    )[0]
    context.log.info(pred_onx)
    return pred_onx
```

### 파티션 사용하기

통합은 기본적으로 [Dagster 파티션](https://docs.dagster.io/concepts/partitions-schedules-sensors/partitions)을 지원합니다.

단일, 다수 또는 모든 자산의 파티션을 선택적으로 읽을 수 있습니다.

모든 파티션은 파티션 키와 아티팩트 콘텐츠를 나타내는 키와 값으로 구성된 사전에서 제공됩니다.

<Tabs
  defaultValue="all"
  values={[
    {label: '모든 파티션 읽기', value: 'all'},
    {label: '특정 파티션 읽기', value: 'specific'},
  ]}>
  <TabItem value="all">

상류 `@asset`의 모든 파티션을 읽습니다. 이 사전에서 키와 값은 각각 파티션 키와 아티팩트 콘텐츠와 관련됩니다.
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

`AssetIn`의 `partition_mapping` 구성을 사용하면 특정 파티션을 선택할 수 있습니다. 이 경우, `TimeWindowPartitionMapping`을 사용하고 있습니다.
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

구성 객체 `metadata`는 프로젝트의 다른 아티팩트 파티션과 Weights & Biases(wandb)가 어떻게 상호작용하는지 구성하는 데 사용됩니다.

`metadata` 객체는 `wandb_artifact_configuration`라는 키를 포함하며 이는 또한 `partitions`라는 중첩 객체를 포함합니다.

`partitions` 객체는 각 파티션의 이름을 해당 구성에 매핑합니다. 각 파티션의 구성은 그로부터 데이터를 검색하는 방법을 지정할 수 있습니다. 이러한 구성에는 `get`, `version`, `alias`와 같은 다양한 키가 포함될 수 있으며, 각 파티션의 요구사항에 따라 다릅니다.

**구성 키**

1. `get`:
`get` 키는 데이터를 가져올 W&B 개체(테이블, 이미지...)의 이름을 지정합니다.
2. `version`:
`version` 키는 아티팩트의 특정 버전을 가져오려고 할 때 사용됩니다.
3. `alias`:
`alias` 키를 사용하면 아티팩트를 별칭으로 가져올 수 있습니다.

**와일드카드 구성**

와일드카드 `"*"`는 구성되지 않은 모든 파티션을 나타냅니다. 이는 `partitions` 객체에서 명시적으로 언급되지 않은 파티션에 대한 기본 구성을 제공합니다.

예를 들어,

```python
"*": {
    "get": "default_table_name",
},
```
이 구성은 명시적으로 구성되지 않은 모든 파티션에 대해 `default_table_name`이라는 테이블에서 데이터를 가져온다는 의미입니다.

**특정 파티션 구성**

특정 파티션의 구성을 제공하여 와일드카드 구성을 재정의할 수 있습니다.

예를 들어,

```python
"yellow": {
    "get": "custom_table_name",
},
```

이 구성은 `yellow`라는 이름의 파티션에 대해 `custom_table_name`이라는 테이블에서 데이터를 가져온다는 의미이며, 와일드카드 구성을 재정의합니다.

**버전 관리 및 별칭 사용**

버전 관리 및 별칭 사용 목적으로 구성에 특정 `version` 및 `alias` 키를 제공할 수 있습니다.

버전의 경우,

```python
"orange": {
    "version": "v0",
},
```

이 구성은 `orange` 아티팩트 파티션의 버전 `v0`에서 데이터를 가져옵니다.

별칭의 경우,

```python
"blue": {
    "alias": "special_alias",
},
```

이 구성은 별칭 `special_alias`로 참조되는 아티팩트 파티션(구성에서 `blue`로 언급됨)의 `default_table_name` 테이블에서 데이터를 가져옵니다.

### 고급 사용법
통합의 고급 사용법을 보려면 다음 전체 코드 예제를 참조하십시오:
* [자산에 대한 고급 사용 예제](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/assets/advanced_example.py)
* [파티션된 작업 예제](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/ops/partitioned_job.py)
* [모델을 모델 레지스트리에 연결하기](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/assets/model_registry_example.py)

## W&B Launch 사용하기

:::caution
활발한 개발 중인 베타 제품
Launch에 관심이 있으신가요? W&B Launch 고객 파일럿 프로그램에 참여하기 위해 계정 팀에 문의하십시오.
파일럿 고객은 AWS EKS 또는 SageMaker를 사용해야 베타 프로그램에 참여할 자격이 있습니다. 최종적으로 추가 플랫폼을 지원할 계획입니다.
:::

계속하기 전에, W&B Launch 사용 방법을 잘 이해하는 것이 좋습니다. Launch에 대한 가이드를 읽어보십시오: https://docs.wandb.ai/guides/launch.

Dagster 통합은 다음과 같은 도움을 줍니다:
* Dagster 인스턴스에서 하나 이상의 Launch 에이전트를 실행합니다.
* Dagster 인스턴스 내에서 로컬 Launch 작업을 실행합니다.
* 온-프레미스 또는 클라우드에서 원격 Launch 작업을 실행합니다.

### Launch 에이전트
통합은 `run_launch_agent`라는 이름의 가져올 수 있는 `@op`를 제공합니다. 이것은 Launch 에이전트를 시작하고 수동으로 중지될 때까지 장기 실행 프로세스로 실행합니다.

에이전트는 launch 큐를 폴링하고 작업을 순서대로 실행하거나(또는 외부 서비스에 디스패치하여 실행하도록) 하는 프로세스입니다.

구성에 대해서는 [참조 문서](../launch/intro.md)를 참조하십시오.

Launchpad에서 모든 속성에 대한 유용한 설명도 볼 수 있습니다.

![](/images/integrations/dagster_launch_agents.png)

간단한 예제
```python
# config.yaml에 이것을 추가합니다
# 또는 Dagit의 Launchpad 또는 JobDefinition.execute_in_process에서 구성을 설정할 수 있습니다
# 참조: https://docs.dagster.io/concepts/configuration/config-schema#specifying-runtime-configuration
resources:
 wandb_config:
   config:
     entity: my_entity # 이것을 귀하의 W&B 엔티티로 교체하십시오
     project: my_project # 이것을 귀하의 W&B 프로젝트로 교체하십시오
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
통합은 `run_launch_job`이라는 이름의 가져올 수 있는 `@op`를 제공합니다. 이것은 Launch 작업을 실행합니다.

Launch 작업은 실행될 수 있도록 큐에 할당됩니다. 큐를 생성하거나 기본 큐를 사용할 수 있습니다. 해당 큐를 청취하는 활성 에이전트가 있는지 확인하십시오. Dagster 인스턴스 내에서 에이전트를 실행할 수 있지만, Kubernetes에서 배포 가능한 에이전트를 사용하는 것도 고려할 수 있습니다.

구성에 대해서는 [참조 문서](../launch/intro.md)를 참조하십시오.

Launchpad에서 모든 속성에 대한 유용한 설명도 볼 수 있습니다.

![](/images/integrations/dagster_launch_jobs.png)

간단한 예제
```python
# config.yaml에 이것을 추가합니다
# 또는 Dagit의 Launchpad 또는 JobDefinition.execute_in_process에서 구성을 설정할 수 있습니다
# 참조: https://docs.dagster.io/concepts/configuration/config-schema#specifying-runtime-configuration
resources:
 wandb_config:
   config:
     entity: my_entity # 이것을 귀하의 W&B 엔티티로 교체하십시오
     project: my_project # 이것을 귀하의 W&B 프로젝트로 교체하십시오
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

from dagster는 job, make_values_resource


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
   run_launch_job.alias("my_launched_job")() # 작업을 별칭으로 이름을 바꿉니다
```

## 모범 사례

1. 아티팩트를 읽고 쓰기 위해 IO 관리자를 사용하십시오.
[`Artifact.download()`](../../ref/python/artifact.md#download) 또는 [`Run.log_artifact()`](../../ref/python/run.md#log_artifact)를 직접 사용할 필요가 없어야 합니다. 이러한 메서드는 통합에 의해 처리됩니다. 단순히 아티팩트에 저장하고자 하는 데이터를 반환하고 나머지는 통합이 처리하도록 하십시오. 이렇게 하면 W&B에서 아티팩트의 계보가 더 잘 제공됩니다.

2. 복잡한 사용 사례를 위해 직접 아티팩트 개체를 구성하십시오.
Python 개체와 W&B 개체는 ops/assets에서 반환되어야 합니다. 통합은 아티팩트를 번들링합니다.
복잡한 사용 사례의 경우, Dagster 작업에서 직접 아티팩트를 구성할 수 있습니다. 메타데이터 풍부화를 위해 통합에 아티팩트 개체를 전달하는 것이 좋습니다. 예를 들어, 소스
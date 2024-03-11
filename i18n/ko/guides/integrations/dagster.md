---
description: Guide on how to integrate W&B with Dagster.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Dagster

Dagster와 W&B(W&B)를 사용하여 MLOps 파이프라인을 조율하고 ML 자산을 관리하세요. Dagster와의 통합을 통해 다음 작업을 쉽게 수행할 수 있습니다:

* [W&B Artifacts](../artifacts/intro.md) 사용 및 생성.
* [W&B Model Registry](../model_registry/intro.md)에서 등록된 모델 사용 및 생성.
* [W&B Launch](../launch/intro.md)를 사용하여 전용 컴퓨팅에서 트레이닝 작업 실행.
* ops 및 자산에서 [wandb](../../ref/python/README.md) 클라이언트 사용.

W&B Dagster 통합은 W&B 전용 Dagster 리소스 및 IO 관리자를 제공합니다:

* `wandb_resource`: W&B API에 인증하고 통신하는 데 사용되는 Dagster 리소스입니다.
* `wandb_artifacts_io_manager`: W&B Artifacts를 사용하는 데 사용되는 Dagster IO 관리자입니다.

다음 가이드는 Dagster에서 W&B를 사용하기 위한 전제 조건을 충족하는 방법, ops 및 자산에서 W&B Artifacts를 생성하고 사용하는 방법, W&B Launch를 사용하는 방법 및 권장 모범 사례를 설명합니다.

## 시작하기 전에
Weights and Biases 내에서 Dagster를 사용하려면 다음 리소스가 필요합니다:
1. **W&B API 키**.
2. **W&B 엔터티(사용자 또는 팀)**: 엔터티는 W&B Runs 및 Artifacts를 보내는 사용자 이름 또는 팀 이름입니다. W&B App UI에서 계정 또는 팀 엔터티를 만들고 실행을 기록하기 전에 확인하세요. 엔터티를 지정하지 않으면 실행은 일반적으로 사용자 이름인 기본 엔터티로 보내집니다. 설정에서 **프로젝트 기본값** 아래에서 기본 엔터티를 변경하세요.
3. **W&B 프로젝트**: [W&B Runs](../runs/intro.md)가 저장되는 프로젝트의 이름입니다.

W&B 앱에서 해당 사용자 또는 팀의 프로필 페이지를 확인하여 W&B 엔터티를 찾을 수 있습니다. 기존 W&B 프로젝트를 사용하거나 새로 만들 수 있습니다. 새 프로젝트는 W&B 앱 홈페이지 또는 사용자/팀 프로필 페이지에서 생성할 수 있습니다. 프로젝트가 없으면 처음 사용할 때 자동으로 생성됩니다. 다음 지침은 API 키를 얻는 방법을 보여줍니다:

### API 키 얻는 방법
1. [W&B에 로그인](https://wandb.ai/login)하세요. 참고: W&B 서버를 사용하는 경우 관리자에게 인스턴스 호스트 이름을 문의하세요.
2. [인증 페이지](https://wandb.ai/authorize)로 이동하거나 사용자/팀 설정에서 API 키를 수집하세요. 프로덕션 환경에서는 해당 키를 소유하는 [서비스 계정](https://docs.wandb.ai/guides/technical-faq/general#what-is-a-service-account-and-why-is-it-useful)을 사용하는 것이 좋습니다.
3. 해당 API 키에 대한 환경 변수를 설정하세요 export `WANDB_API_KEY=YOUR_KEY`.

다음 예제는 Dagster 코드에서 API 키를 지정하는 위치를 보여줍니다. `wandb_config` 중첩 딕셔너리 내에서 엔터티와 프로젝트 이름을 지정해야 합니다. 다른 W&B 프로젝트를 사용하려면 다른 ops/자산에 다른 `wandb_config` 값을 전달할 수 있습니다. 전달할 수 있는 키에 대한 자세한 정보는 아래 설정 섹션을 참조하세요.

<Tabs
  defaultValue="job"
  values={[
    {label: 'configuration for @job', value: 'job'},
    {label: 'configuration for @repository using assets', value: 'repository'},
  ]}>
  <TabItem value="job">

예시: `@job`에 대한 설정
```python
# config.yaml에 이를 추가하세요
# 또는 Dagit의 Launchpad 또는 JobDefinition.execute_in_process에서 설정을 설정할 수 있습니다
# 참조: https://docs.dagster.io/concepts/configuration/config-schema#specifying-runtime-configuration
resources:
 wandb_config:
   config:
     entity: my_entity # 이를 W&B 엔터티로 바꾸세요
     project: my_project # 이를 W&B 프로젝트로 바꾸세요


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


예시: 자산을 사용하는 `@repository`에 대한 설정

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
                       "entity": "my_entity", # 이를 W&B 엔터티로 바꾸세요
                       "project": "my_project", # 이를 W&B 프로젝트로 바꾸세요
                   }
               }
           },
       ),
   ]
```
이 예시에서는 `@job`에 대한 예시와 반대로 IO 관리자 캐시 기간을 구성합니다.

  </TabItem>
</Tabs>

### 설정
다음 설정 옵션은 통합에 제공된 W&B 전용 Dagster 리소스 및 IO 관리자에 대한 설정으로 사용됩니다.

* `wandb_resource`: W&B API와 통신하는 데 사용되는 Dagster [리소스](https://docs.dagster.io/concepts/resources)입니다. 제공된 API 키를 사용하여 자동으로 인증합니다. 속성:
    * `api_key`: (str, 필수): W&B API와 통신하는 데 필요한 W&B API 키입니다.
    * `host`: (str, 선택사항): 사용하려는 API 호스트 서버입니다. W&B 서버를 사용하는 경우에만 필요합니다. 기본값은 공용 클라우드 호스트입니다: [https://api.wandb.ai](https://api.wandb.ai)
* `wandb_artifacts_io_manager`: W&B Artifacts를 사용하는 데 사용되는 Dagster [IO 관리자](https://docs.dagster.io/concepts/io-management/io-managers)입니다. 속성:
    * `base_dir`: (int, 선택사항) 로컬 저장소 및 캐싱에 사용되는 기본 디렉토리입니다. W&B Artifacts 및 W&B Run 로그는 해당 디렉토리에서 읽고 쓰입니다. 기본적으로 `DAGSTER_HOME` 디렉토리를 사용합니다.
    * `cache_duration_in_minutes`: (int, 선택사항) W&B Artifacts 및 W&B Run 로그를 로컬 저장소에 보관해야 하는 시간을 정의합니다. 지정된 시간 동안 열리지 않은 파일 및 디렉토리만 캐시에서 제거됩니다. 캐시 정리는 IO 관리자 실행이 끝날 때 발생합니다. 캐싱을 완전히 비활성화하려면 0으로 설정할 수 있습니다. 캐싱은 동일한 기계에서 실행되는 작업 간에 아티팩트가 재사용될 때 속도를 향상시킵니다. 기본값은 30일입니다.
    * `run_id`: (str, 선택사항): 이 실행에 대한 고유 ID로, 재개에 사용됩니다. 프로젝트에서 고유해야 하며, 실행을 삭제하면 ID를 재사용할 수 없습니다. 짧은 설명 이름에는 이름 필드를 사용하거나, 실행을 비교하는 데 하이퍼파라미터를 저장하는 데 config를 사용하세요. ID에는 다음 특수 문자를 포함할 수 없습니다: `/\#?%:..` Dagster 내에서 실험 추적을 수행할 때 실행 ID를 설정해야 IO 관리자가 실행을 재개할 수 있습니다. 기본적으로 Dagster 실행 ID로 설정됩니다. 예: `7e4df022-1bf2-44b5-a383-bb852df4077e`.
    * `run_name`: (str, 선택사항) 이 실행에 대한 짧은 표시 이름으로, UI에서 이 실행을 식별하는 데 사용됩니다. 기본적으로 다음 형식의 문자열로 설정됩니다. dagster-run-[Dagster 실행 ID의 처음 8자] 예: `dagster-run-7e4df022`.
    * `run_tags`: (list[str], 선택사항): UI에서 이 실행의 태그 목록에 추가될 문자열 목록입니다. 태그는 실행을 함께 구성하거나 "베이스라인" 또는 "프로덕션"과 같은 임시 레이블을 적용하는 데 유용합니다. UI에서 태그를 추가하거나 제거하거나 특정 태그가 있는 실행만 필터링하는 것이 쉽습니다. 통합에서 사용하는 모든 W&B 실행에는 `dagster_wandb` 태그가 있습니다.

## W&B Artifacts 사용하기

W&B Artifact와의 통합은 Dagster IO 관리자에 의존합니다.

[IO 관리자](https://docs.dagster.io/concepts/io-management/io-managers)는 자산 또는 op의 출력을 저장하고 다운스트림 자산 또는 op에 입력으로 로드하는 책임이 있는 사용자 제공 객체입니다. 예를 들어, IO 관리자는 파일 시스템의 파일에서 객체를 저장하고 로드할 수 있습니다.

통합은 W&B Artifacts에 대한 IO 관리자를 제공합니다. 이를 통해 Dagster `@op` 또는 `@asset`이 W&B Artifacts를 기본적으로 생성하고 사용할 수 있습니다. 여기 Python 리스트가 포함된 데이터셋 유형의 W&B Artifact를 생성하는 `@asset`의 간단한 예가 있습니다.

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
    return [1, 2, 3] # 이것은 아티팩트에 저장됩니다
```

Artifacts를 작성하기 위해 메타데이터 구성으로 `@op`, `@asset` 및 `@multi_asset`을 주석 처리할 수 있습니다. 마찬가지로 Dagster 외부에서 생성된 W&B Artifacts도 사용할 수 있습니다.

## W&B Artifacts 작성하기
계속하기 전에 W&B Artifacts 사용 방법을 잘 이해하는 것이 좋습니다. [Artifacts에 대한 가이드](../artifacts/intro.md)를 읽어보세요.

Python 함수에서 객체를 반환하여 W&B Artifact를 작성하세요. W&B에서 지원하는 다음 객체는 다음과 같습니다:
* Python 객체 (int, dict, list…)
* W&B 객체 (Table, Image, Graph…)
* W&B Artifact 객체

다음 예제는 Dagster 자산(`@asset`)으로 W&B Artifacts를 작성하는 방법을 보여줍니다:

<Tabs
  defaultValue="python_objects"
  values={[
    {label: 'Python objects', value: 'python_objects'},
    {label: 'W&B object', value: 'wb_object'},
    {label: 'W&B Artifacts', value: 'wb_artifact'},
  ]}>
  <TabItem value="python_objects">

[pickle](https://docs.python.org/3/library/pickle.html) 모듈로 직렬화할 수 있는 모든 것은 통합에 의해 생성된 Artifact에 피클링되어 추가됩니다. Dagster 내부에서 해당 Artifact를 읽을 때 내용은 unpickling됩니다(자세한 내용은 [Read artifacts](#read-wb-artifacts) 참조).

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


W&B는 여러 Pickle 기반 직렬화 모듈([pickle](https://docs.python.org/3/library/pickle.html), [dill](https://github.com/uqfoundation/dill), [cloudpickle](https://github.com/cloudpipe/cloudpickle), [joblib](https://github.com/joblib/joblib))을 지원합니다. 또한 [ONNX](https://onnx.ai/) 또는 [PMML](https://en.wikipedia.org/wiki/Predictive_Model_Markup_Language)과 같은 보다 고급 직렬화를 사용할 수도 있습니다. 자세한 정보는 [직렬화](#serialization-configuration) 섹션을 참조하세요.

  </TabItem>
  <TabItem value="wb_object">

기본 W&B 객체(예: [Table](../../ref/python/data-types/table.md), [Image](../../ref/python/data-types/image.md), [Graph](../../ref/python/data-types/graph.md))는 통합에 의해 생성된 Artifact에 추가됩니다. 여기 Table을 사용한 예가 있습니다.

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

복잡한 사용 사례의 경우, 자체 Artifact 객체를 빌드해야 할 수도 있습니다. 통합은 여전히 통합 양쪽의 메타데이터를 확장하는 유용한 추가 기능을 제공합니다.

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

### 설정
`@op`, `@asset` 및 `@multi_asset`의 데코레이터 인수로 메타데이터에 전달해야 하는 설정 딕셔너리인 wandb_artifact_configuration을 설정할 수 있습니다. 이 설정은 W&B Artifacts의 IO 관리자 읽기 및 쓰기를 제어하는 데 필요합니다.

`@op`의 경우 출력 메타데이터를 통해 [Out](https://docs.dagster.io/_apidocs/ops#dagster.Out) 메타데이터 인수에 있습니다.
`@asset`의 경우 자산의 메타데이터 인수에 있

#### 고급 사용법
- [파티션 작업](https://github.com/dagster-io/dagster/blob/master/examples/with_wandb/with_wandb/ops/partitioned_job.py)
- [간단한 파티션 자산](https://github.com/wandb/dagster/blob/master/examples/with_wandb/with_wandb/assets/simple_partitions_example.py)
- [다중 파티션 자산](https://github.com/wandb/dagster/blob/master/examples/with_wandb/with_wandb/assets/multi_partitions_example.py)
- [고급 파티션 사용법](https://github.com/wandb/dagster/blob/master/examples/with_wandb/with_wandb/assets/advanced_partitions_example.py)

## Weights & Biases 아티팩트 읽기
Weights & Biases 아티팩트를 읽는 것은 작성하는 것과 유사합니다. `@op` 또는 `@asset`에 `wandb_artifact_configuration`이라는 설정 사전을 설정할 수 있습니다. 유일한 차이점은 설정을 출력 대신 입력에 설정해야 한다는 것입니다.

`@op`의 경우, 입력 메타데이터를 통해 [In](https://docs.dagster.io/_apidocs/ops#dagster.In) 메타데이터 인수를 통해 위치합니다. 아티팩트의 이름을 명시적으로 전달해야 합니다.

`@asset`의 경우, 입력 메타데이터를 통해 [Asset](https://docs.dagster.io/_apidocs/assets#dagster.AssetIn) In 메타데이터 인수를 통해 위치합니다. 부모 자산의 이름이 일치해야 하므로 아티팩트 이름을 전달해서는 안 됩니다.

인테그레이션 외부에서 생성된 아티팩트에 대한 의존성이 있는 경우 [SourceAsset](https://docs.dagster.io/_apidocs/assets#dagster.SourceAsset)을 사용해야 합니다. 항상 해당 자산의 최신 버전을 읽습니다.

다음 예제는 다양한 ops에서 아티팩트를 읽는 방법을 보여줍니다.

<Tabs
  defaultValue="op"
  values={[
    {label: '@op에서', value: 'op'},
    {label: '다른 @asset에 의해 생성된', value: 'asset'},
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
           # 입력 인수의 이름을 변경하고 싶지 않다면 'key'를 제거할 수 있습니다
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

### 설정
진행 중인 설정은 IO 매니저가 장식된 함수에 입력으로 수집하고 제공해야 하는 것을 나타냅니다. 다음 읽기 패턴이 지원됩니다.

1. 아티팩트 내에 포함된 명명된 오브젝트를 가져오려면 get을 사용하세요:

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


2. 아티팩트 내에 포함된 다운로드된 파일의 로컬 경로를 가져오려면 get_path를 사용하세요:

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

3. 내용이 로컬로 다운로드된 아티팩트 객체 전체를 가져오려면:
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
* `get`: (str) 아티팩트 상대 이름에서 W&B 오브젝트(테이블, 이미지...)를 가져옵니다.
* `get_path`: (str) 아티팩트 상대 이름에서 파일로 가는 경로를 가져옵니다.

### 직렬화 설정
기본적으로 통합은 표준 [pickle](https://docs.python.org/3/library/pickle.html) 모듈을 사용하지만, 일부 오브젝트는 이와 호환되지 않습니다. 예를 들어, yield가 있는 함수는 pickle하려고 하면 오류가 발생할 수 있습니다.

우리는 더 많은 Pickle 기반 직렬화 모듈([dill](https://github.com/uqfoundation/dill), [cloudpickle](https://github.com/cloudpipe/cloudpickle), [joblib](https://github.com/joblib/joblib))을 지원합니다. 또한 [ONNX](https://onnx.ai/) 또는 [PMML](https://en.wikipedia.org/wiki/Predictive_Model_Markup_Language)과 같은 더 고급 직렬화를 사용하여 직렬화된 문자열을 반환하거나 직접 아티팩트를 생성함으로써 사용할 수 있습니다. 적절한 선택은 사용 사례에 따라 달라지므로 이 주제에 관한 기존 문헌을 참조하십시오.

### Pickle 기반 직렬화 모듈

:::caution
Pickle은 보안에 취약한 것으로 알려져 있습니다. 보안이 우려된다면 W&B 오브젝트만 사용하십시오. 데이터에 서명하고 해시 키를 자체 시스템에 저장하는 것이 좋습니다. 더 복잡한 사용 사례에 대해서는 주저하지 말고 문의해 주세요. 도움을 드리겠습니다.
:::

`wandb_artifact_configuration`의 `serialization_module` 사전을 통해 사용할 직렬화를 설정할 수 있습니다. 모듈이 Dagster를 실행하는 기계에 사용 가능한지 확인하십시오.

통합은 해당 아티팩트를 읽을 때 어떤 직렬화 모듈을 사용해야 하는지 자동으로 알게 됩니다.

현재 지원되는 모듈은 pickle, dill, cloudpickle 및 joblib입니다.

joblib로 직렬화된 "모델"을 생성한 다음 추론에 사용하는 간단한 예제입니다.

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
    # 이것은 실제 ML 모델이 아니지만 pickle 모듈로는 불가능할 것입니다
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
    context.log.info(inference_result)  # 출력: 3
    return inference_result
```

### 고급 직렬화 형식 (ONNX, PMML)
ONNX 및 PMML과 같은 교환 파일 형식을 사용하는 것이 일반적입니다. 통합은 이러한 형식을 지원하지만 Pickle 기반 직렬화보다 약간 더 많은 작업이 필요합니다.

이러한 형식을 사용하는 데에는 두 가지 다른 방법이 있습니다.
1. 모델을 선택한 형식으로 변환한 다음, 일반 Python 오브젝트처럼 해당 형식의 문자열 표현을 반환합니다. 통합은 해당 문자열을 pickle할 것입니다. 그런 다음 해당 문자열을 사용하여 모델을 다시 구축할 수 있습니다.
2. 직렬화된 모델로 새로운 로컬 파일을 생성한 다음, 해당 파일을 사용하여 add_file 구성을 사용하여 사용자 정의 아티팩트를 구축합니다.

ONNX를 사용하여 직렬화된 Scikit-learn 모델의 예제입니다.

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
    # https://onnx.ai/sklearn-onnx/에서 영감을 받았습니다

    # 모델 훈련
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clr = RandomForestClassifier()
    clr.fit(X_train, y_train)

    # ONNX 형식으로 변환
    initial_type = [("float_input", FloatTensorType([None, 4]))]
    onx = convert_sklearn(clr, initial_types=initial_type)

    # 아티팩트(모델 + 테스트 세트) 작성
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
    # https://onnx.ai/sklearn-onnx/에서 영감을 받았습니다

    # ONNX Runtime을 사용하여 예측 계산
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

통합은 [Dagster 파티션](https://docs.dagster.io/concepts/partitions-schedules-sensors/partitions)을 기본적으로 지원합니다.

자산의 파티션 하나, 여러 개 또는 모든 파티션을 선택적으로 읽을 수 있습니다.

모든 파티션은 사전으로 제공되며, 키와 값이 각각 파티션 키와 아티팩트 내용을 나타냅니다.



<Tabs
  defaultValue="all"
  values={[
    {label: '모든 파티션 읽기', value: 'all'},
    {label: '특정 파티션 읽기', value: 'specific'},
  ]}>
  <TabItem value="all">

상류 `@asset`의 모든 파티션을 읽습니다. 이 사전에서 키와 값은 각각 파티션 키와 아티팩트 내용을 나타냅니다.
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

`AssetIn`의 `partition_mapping` 구성을 통해 특정 파티션을 선택할 수 있습니다. 이 경우에는 `TimeWindowPartitionMapping`을 사용하고 있습니다.
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

설정 객체인 `metadata`는 Weights & Biases (wandb)가 프로젝트의 다양한 아티팩트 파티션과 상호 작용하는 방식을 구성하는 데 사용됩니다.

`metadata` 객체는 `wandb_artifact_configuration`이라는 키를 포함하며, 이 키는 `partitions`이라는 중첩된 객체를 더 포함합니다.

`partitions` 객체는 각 파티션의 이름을 해당 구성에 매핑합니다. 각 파티션에 대한 구성은 그로부터 데이터를 검색하는 방법을 지정할 수 있습니다. 이러한 구성에는 `get`, `version`, `alias`와 같은 다양한 키가 포함될 수 있으며, 각 파티션의 요구 사항에 따라 다릅니다.

**구성 키**

1. `get`:
`get` 키는 데이터를 가져올 W&B 오브젝트(테이블, 이미지...)의 이름을 지정합니다.
2. `version`:
`version` 키는 아티팩트의 특정 버전을 가져오고자 할 때 사용됩니다.
3. `alias`:
`alias` 키를 사용하면 아티팩트를 그것의 별칭으로 가져올 수 있습니다.

**와일드카드 구성**

와일드카드 `"

## 모범 사례

1. 아티팩트를 읽고 쓰기 위해 IO 매니저를 사용하세요.
[`Artifact.download()`](../../ref/python/artifact.md#download) 또는 [`Run.log_artifact()`](../../ref/python/run.md#log_artifact)을 직접 사용할 필요가 없어야 합니다. 이러한 메소드는 인테그레이션에 의해 처리됩니다. 아티팩트에 저장하고자 하는 데이터를 반환하기만 하면 나머지는 인테그레이션이 처리합니다. 이는 W&B에서 아티팩트의 더 나은 계보를 제공할 것입니다.

2. 복잡한 유스 케이스에만 직접 아티팩트 오브젝트를 구축하세요.
파이썬 오브젝트와 W&B 오브젝트는 여러분의 ops/assets에서 반환되어야 합니다. 인테그레이션은 아티팩트를 번들링합니다.
복잡한 유스 케이스의 경우, Dagster 작업에서 직접 아티팩트를 구축할 수 있습니다. 소스 인테그레이션 이름과 버전, 사용된 파이썬 버전, 피클 프로토콜 버전 등과 같은 메타데이터 풍부화를 위해 인테그레이션에 아티팩트 오브젝트를 전달하는 것이 좋습니다.

3. 아티팩트에 파일, 디렉토리 및 외부 참조를 메타데이터를 통해 추가하세요.
`wandb_artifact_configuration` 오브젝트를 사용하여 파일, 디렉토리 또는 외부 참조(Amazon S3, GCS, HTTP…)를 추가하세요. 자세한 정보는 [아티팩트 설정 섹션](#configuration-1)의 고급 예제를 참조하세요.

4. 아티팩트가 생성될 때 @op 대신 @asset을 사용하세요.
아티팩트는 자산입니다. Dagster가 해당 자산을 유지 관리할 때는 자산을 사용하는 것이 좋습니다. 이는 Dagit Asset Catalog에서 더 나은 관찰 가능성을 제공할 것입니다.

5. Dagster 외부에서 생성된 아티팩트를 소비하기 위해 SourceAsset을 사용하세요.
이를 통해 외부에서 생성된 아티팩트를 읽기 위해 인테그레이션을 활용할 수 있습니다. 그렇지 않으면 인테그레이션에 의해 생성된 아티팩트만 사용할 수 있습니다.

6. 대형 모델에 대한 트레이닝을 전용 컴퓨트에서 조정하기 위해 W&B Launch를 사용하세요.
당신은 Dagster 클러스터 내에서 작은 모델을 트레이닝할 수 있고, GPU 노드를 가진 Kubernetes 클러스터에서 Dagster를 실행할 수 있습니다. 대형 모델 트레이닝에는 W&B Launch를 사용하는 것이 좋습니다. 이는 여러분의 인스턴스를 과부하시키지 않고 더 적절한 컴퓨트에 엑세스할 수 있도록 할 것입니다.

7. Dagster 내에서 실험 추적을 할 때, W&B Run ID를 여러분의 Dagster Run ID 값으로 설정하세요.
우리는 [Run을 이어갈 수 있도록](../runs/resuming.md) 만들고 W&B Run ID를 Dagster Run ID 또는 여러분이 선택한 문자열로 설정하는 것을 추천합니다. 이 권장 사항을 따르면 Dagster 내에서 모델을 트레이닝할 때 여러분의 W&B 메트릭과 W&B 아티팩트가 동일한 W&B Run에 저장됩니다.

Dagster Run ID로 W&B Run ID를 설정하세요.
```python
wandb.init(
    id=context.run_id,
    resume="allow",
    ...
)
```

또는 여러분만의 W&B Run ID를 선택하고 IO 매니저 설정에 전달하세요.
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

8. 큰 W&B 아티팩트를 사용할 때 필요한 데이터만 get 또는 get_path로 수집하세요.
기본적으로 인테그레이션은 전체 아티팩트를 다운로드합니다. 매우 큰 아티팩트를 사용하는 경우 필요한 특정 파일이나 오브젝트만 수집할 수 있습니다. 이는 속도와 리소스 활용을 개선할 것입니다.

9. 파이썬 오브젝트의 경우 유스 케이스에 맞게 피클링 모듈을 조정하세요.
기본적으로 W&B 인테그레이션은 표준 [pickle](https://docs.python.org/3/library/pickle.html) 모듈을 사용합니다. 그러나 일부 오브젝트는 이와 호환되지 않습니다. 예를 들어, yield를 가진 함수는 피클링을 시도할 때 오류를 발생시킬 것입니다. W&B는 다른 피클 기반 직렬화 모듈([dill](https://github.com/uqfoundation/dill), [cloudpickle](https://github.com/cloudpipe/cloudpickle), [joblib](https://github.com/joblib/joblib))을 지원합니다.

직렬화된 문자열을 반환하거나 직접 아티팩트를 생성함으로써 더 고급 직렬화([ONNX](https://onnx.ai/) 또는 [PMML](https://en.wikipedia.org/wiki/Predictive_Model_Markup_Language))을 사용할 수도 있습니다. 올바른 선택은 여러분의 유스 케이스에 따라 달라질 것이며, 이 주제에 관한 사용 가능한 문헌을 참조하세요.
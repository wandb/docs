---
title: Manage job inputs
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

Launch의 핵심 경험은 하이퍼파라미터와 데이터셋 같은 다양한 작업 입력을 손쉽게 실험하고 해당 작업을 적절한 하드웨어로 라우팅하는 것입니다. 작업이 생성되면, 원본 작성자 외의 사용자도 W&B GUI 또는 CLI를 통해 이러한 입력값을 조정할 수 있습니다. CLI 또는 UI에서 작업 입력을 설정하는 방법에 대한 정보는 [Enqueue jobs](./add-job-to-queue.md) 가이드를 참조하세요.

이 섹션에서는 작업의 입력값을 프로그래밍적으로 조정하는 방법을 설명합니다.

기본적으로 W&B 작업은 전체 `Run.config`를 작업의 입력으로 캡처하지만, Launch SDK는 run config에서 선택한 키를 제어하거나 JSON 또는 YAML 파일을 입력으로 지정하는 함수를 제공합니다.

:::info
Launch SDK 함수는 `wandb-core`가 필요합니다. 자세한 내용은 [`wandb-core` README](https://github.com/wandb/wandb/blob/main/core/README.md)를 참조하세요.
:::

## `Run` 오브젝트 재설정

작업 내에서 `wandb.init`에 의해 반환된 `Run` 오브젝트는 기본적으로 재설정할 수 있습니다. Launch SDK는 작업을 시작할 때 어떤 부분의 `Run.config` 오브젝트를 재설정할 수 있는지 사용자 정의할 수 있는 방법을 제공합니다.

```python
import wandb
from wandb.sdk import launch

# Launch sdk 사용을 위해 필요합니다.
wandb.require("core")

config = {
    "trainer": {
        "learning_rate": 0.01,
        "batch_size": 32,
        "model": "resnet",
        "dataset": "cifar10",
        "private": {
            "key": "value",
        },
    },
    "seed": 42,
}


with wandb.init(config=config):
    launch.manage_wandb_config(
        include=["trainer"], 
        exclude=["trainer.private"],
    )
    # 기타.
```

`launch.manage_wandb_config` 함수는 `Run.config` 오브젝트에 대한 입력 값을 수락하도록 작업을 구성합니다. 선택적 옵션인 `include` 및 `exclude`는 중첩된 설정 오브젝트 내의 경로 접두사를 가져옵니다. 예를 들어, 작업이 사용자에게 노출하고 싶지 않은 옵션을 가진 라이브러리를 사용할 때 유용할 수 있습니다.

`include` 접두사가 제공되면, 설정 내에서 `include` 접두사와 일치하는 경로만 입력 값을 수락합니다. `exclude` 접두사가 제공되면, `exclude` 목록과 일치하는 경로는 입력 값에서 필터링됩니다. 경로가 `include` 및 `exclude` 접두사 모두와 일치하는 경우, `exclude` 접두사가 우선합니다.

앞의 예에서, 경로 `["trainer.private"]`는 `trainer` 오브젝트에서 `private` 키를 필터링하고, 경로 `["trainer"]`는 `trainer` 오브젝트 아래에 없는 모든 키를 필터링합니다.

:::tip
`.`이름에 있는 키를 필터링하려면 `.`이스케이프된 `\`를 사용하세요.

예를 들어, `r"trainer\.private"`는 `trainer.private` 키를 필터링하며, 이는 `trainer` 오브젝트 아래의 `private` 키와 다릅니다.

위의 `r` 접두는 원문 문자열을 나타냄에 유의하세요.
:::

위 코드가 패키징되어 작업으로 실행되면, 작업의 입력 유형은 다음과 같습니다:

```json
{
    "trainer": {
        "learning_rate": "float",
        "batch_size": "int",
        "model": "str",
        "dataset": "str",
    },
}
```

W&B CLI 또는 UI에서 작업을 실행할 때, 사용자는 오직 네 가지 `trainer` 파라미터만 오버라이드할 수 있습니다.

### run config 입력값 엑세스

run config 입력값으로 실행된 작업은 `Run.config`를 통해 입력값에 엑세스할 수 있습니다. 작업 코드 내에서 `wandb.init`에 의해 반환된 `Run`은 입력값이 자동으로 설정됩니다. 작업 코드 어디서든 다음을 사용하세요.
```python
from wandb.sdk import launch

run_config_overrides = launch.load_wandb_config()
```
작업 코드 내에서 run config 입력값을 로드하려면 사용하세요.

## 파일 재설정

Launch SDK는 작업 코드 내의 설정 파일에 저장된 입력값을 관리할 수 있는 방법도 제공합니다. 이는 많은 딥러닝 및 대형 언어 모델 유스 케이스, 예를 들어 이 [torchtune](https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama3/8B_lora.yaml) 예제나 이 [Axolotl config](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/llama-3/qlora-fsdp-70b.yaml)와 같은 경우에 흔합니다.

:::info
[Sweeps on Launch](./sweeps-on-launch.md)는 설정 파일 입력을 스윕 파라미터로 사용하는 것을 지원하지 않습니다. 스윕 파라미터는 `Run.config` 오브젝트를 통해 제어해야 합니다.
:::

`launch.manage_config_file` 함수는 Launch 작업에 입력으로 설정 파일을 추가하여 작업 실행 시 설정 파일 내 값을 편집할 수 있도록 합니다.

기본적으로, `launch.manage_config_file`을 사용하면 run config 입력값이 캡처되지 않습니다. `launch.manage_wandb_config`을 호출하면 이 행동을 무시합니다.

다음 예제를 고려하세요:

```python
import yaml
import wandb
from wandb.sdk import launch

# Launch sdk 사용을 위해 필요합니다.
wandb.require("core")

launch.manage_config_file("config.yaml")

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

with wandb.init(config=config):
    # 기타.
    pass
```

코드가 `config.yaml`이라는 인접 파일과 함께 실행된다고 상상해 보세요:

```yaml
learning_rate: 0.01
batch_size: 32
model: resnet
dataset: cifar10
```

`launch.manage_config_file` 호출은 `config.yaml` 파일을 작업에 입력으로 추가하여, W&B CLI나 UI에서 작업을 실행할 때 다시 구성할 수 있게 만듭니다.

`include` 및 `exclude` 키워드 인수를 사용하여 `launch.manage_wandb_config`와 동일한 방식으로 설정 파일의 허용 가능한 입력 키를 필터링할 수 있습니다.

### 설정 파일 입력값 엑세스

Launch에 의해 생성된 run에서 `launch.manage_config_file`이 호출되면 `launch`는 설정 파일의 내용을 입력값으로 패치합니다. 패치된 설정 파일은 작업 환경에서 사용할 수 있습니다.

:::important
작업 코드에서 설정 파일을 읽기 전에 입력값이 사용되도록 `launch.manage_config_file`을 호출하세요.
:::

### 작업의 실행 드로어 UI 사용자 정의

작업의 입력값에 대한 스키마를 정의하면 작업 실행을 위한 사용자 정의 UI를 만들 수 있습니다. 작업의 스키마를 정의하려면 `launch.manage_wandb_config` 또는 `launch.manage_config_file` 호출에 포함하십시오. 스키마는 [JSON Schema](https://json-schema.org/understanding-json-schema/reference)의 형태로 된 파이썬 dict 또는 Pydantic 모델 클래스일 수 있습니다.

:::important
작업 입력 스키마는 입력값을 검증하는 데 사용되지 않습니다. 오직 실행 드로어의 UI를 정의하는 데 사용됩니다.
:::

<Tabs
  defaultValue="jsonschema"
  values={[
    {label: 'JSON Schema', value: 'jsonschema'},
    {label: 'Pydantic Model', value: 'pydantic'},
  ]}>
  <TabItem value="jsonschema">

다음 예제는 이러한 속성을 가진 스키마를 보여줍니다:

- `seed`, 정수
- `trainer`, 일부 키가 지정된 사전:
  - `trainer.learning_rate`, 0보다 커야 하는 부동 소수점
  - `trainer.batch_size`, 16, 64 또는 256 중 하나여야 하는 정수
  - `trainer.dataset`, `cifar10` 또는 `cifar100` 중 하나여야 하는 문자열

```python
schema = {
    "type": "object",
    "properties": {
        "seed": {
          "type": "integer"
        }
        "trainer": {
            "type": "object",
            "properties": {
                "learning_rate": {
                    "type": "number",
                    "description": "모델의 학습률",
                    "exclusiveMinimum": 0,
                },
                "batch_size": {
                    "type": "integer",
                    "description": "배치당 샘플 수",
                    "enum": [16, 64, 256]
                },
                "dataset": {
                    "type": "string",
                    "description": "사용할 데이터셋의 이름",
                    "enum": ["cifar10", "cifar100"]
                }
            }
        }
    }
}

launch.manage_wandb_config(
    include=["seed", "trainer"], 
    exclude=["trainer.private"],
    schema=schema,
)
```

일반적으로, 다음 JSON Schema 속성들이 지원됩니다:

| 속성 | 필수 |  주석 |
| --- | --- | --- |
| `type` | Yes | "number", "integer", "string", "object" 중 하나여야 합니다. |
| `title` | No | 속성의 표시 이름을 재정의합니다. |
| `description` | No | 속성의 도움말 텍스트를 제공합니다. |
| `enum` | No | 자유 형식 텍스트 입력 대신 드롭다운 선택을 만듭니다. |
| `minimum` | No | `type`이 "number" 또는 "integer"일 때만 허용됩니다. |
| `maximum` | No | `type`이 "number" 또는 "integer"일 때만 허용됩니다. |
| `exclusiveMinimum` | No | `type`이 "number" 또는 "integer"일 때만 허용됩니다. |
| `exclusiveMaximum` | No | `type`이 "number" 또는 "integer"일 때만 허용됩니다. |
| `properties` | No | `type`이 "object"일 경우 중첩된 설정을 정의하는 데 사용됩니다. |

  </TabItem>
  <TabItem value="pydantic">

다음 예제는 다음 속성을 가진 스키마를 보여줍니다:

- `seed`, 정수
- `trainer`, 일부 하위 속성이 지정된 스키마:
  - `trainer.learning_rate`, 0보다 커야 하는 부동 소수점
  - `trainer.batch_size`, 1과 256 사이의 값을 가져야 하는 정수, 양단 포함
  - `trainer.dataset`, `cifar10` 또는 `cifar100` 중 하나여야 하는 문자열

```python
class DatasetEnum(str, Enum):
    cifar10 = "cifar10"
    cifar100 = "cifar100"

class Trainer(BaseModel):
    learning_rate: float = Field(gt=0, description="모델의 학습률")
    batch_size: int = Field(ge=1, le=256, description="배치당 샘플 수")
    dataset: DatasetEnum = Field(title="데이터셋", description="사용할 데이터셋의 이름")

class Schema(BaseModel):
    seed: int
    trainer: Trainer

launch.manage_wandb_config(
    include=["seed", "trainer"],
    exclude=["trainer.private"],
    schema=Schema,
)
```

클래스의 인스턴스를 사용할 수도 있습니다:

```python
t = Trainer(learning_rate=0.01, batch_size=32, dataset=DatasetEnum.cifar10)
s = Schema(seed=42, trainer=t)
launch.manage_wandb_config(
    include=["seed", "trainer"],
    exclude=["trainer.private"],
    input_schema=s,
)
```
  </TabItem>
</Tabs>

작업 입력 스키마를 추가하면 실행 드로어 내에서 구조화된 양식을 만들어 작업을 더 쉽게 실행할 수 있습니다.

![](/images/launch/schema_overrides.png)
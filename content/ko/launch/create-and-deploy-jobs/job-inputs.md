---
title: 작업 입력 관리
menu:
  launch:
    identifier: ko-launch-create-and-deploy-jobs-job-inputs
    parent: create-and-deploy-jobs
url: guides/launch/job-inputs
---

Launch 의 핵심 경험은 하이퍼파라미터나 데이터셋 같은 다양한 job 입력값을 손쉽게 실험하고, 이 job 들을 적합한 하드웨어로 라우팅하는 것입니다. 한 번 job 이 생성되면, 원 작성자가 아닌 사용자도 W&B GUI 또는 CLI를 통해 이러한 입력값을 조정할 수 있습니다. CLI나 UI에서 job 입력값을 설정하는 방법에 대한 자세한 내용은 [Enqueue jobs]({{< relref path="./add-job-to-queue.md" lang="ko" >}}) 가이드를 참고하세요.

이 섹션에서는 프로그램적으로 조정 가능한 job 입력값을 제어하는 방법을 설명합니다.

기본적으로, W&B job 은 전체 `Run.config` 를 job 의 입력값으로 캡처하지만, Launch SDK 를 활용하면 run config 내 특정 키를 선별해 입력으로 선택하거나, JSON 혹은 YAML 파일을 입력값으로 지정하는 기능을 제공합니다.

{{% alert %}}
Launch SDK 기능은 `wandb-core` 를 필요로 합니다. 자세한 내용은 [`wandb-core` README](https://github.com/wandb/wandb/blob/main/core/README.md) 를 참고하세요.
{{% /alert %}}

## `Run` 오브젝트 재구성하기

job 에서 `wandb.init` 로 반환된 `Run` 오브젝트는 기본적으로 재구성이 가능합니다. Launch SDK 를 사용하면 job 을 실행할 때 어떤 `Run.config` 오브젝트의 일부만 재구성 가능하도록 커스터마이즈할 수 있습니다.

```python
import wandb
from wandb.sdk import launch

# launch sdk 사용을 위해 필수
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
    # 기타 코드 등등
```

`launch.manage_wandb_config` 함수는 job 이 `Run.config` 오브젝트의 입력값을 받도록 설정합니다. 선택적인 `include`, `exclude` 옵션은 중첩된 config 오브젝트 내에서 경로 prefix 를 지정할 수 있습니다. 예를 들어, job 이 사용자의 노출을 원치 않는 라이브러리의 옵션을 사용한다면 이 설정이 유용합니다.

`include` prefix 를 지정하면, 해당 prefix 와 일치하는 경로에만 입력값을 적용합니다. `exclude` prefix 를 지정했을 경우, 해당 경로는 입력값에서 제외됩니다. 경로가 `include` 와 `exclude` 에 모두 일치한다면, `exclude` 가 우선적으로 적용됩니다.

앞선 예시에서 `["trainer.private"]` 경로는 `trainer` 오브젝트 아래의 `private` 키를 필터링하고, `["trainer"]` 경로는 `trainer` 오브젝트 하위의 키만 허용합니다.

{{% alert %}}
키 이름에 `.` 가 포함되어 있다면 `\`로 이스케이프해서 필터링하세요.

예를 들어, `r"trainer\.private"` 은 `trainer` 오브젝트의 `private` 키가 아닌, `trainer.private` 라는 키를 필터링합니다.

위 코드의 `r` prefix 는 raw string 을 의미합니다.
{{% /alert %}}

위 코드를 job 으로 패키징해 실행하면, job 의 입력 타입은 아래와 같습니다.

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

즉, W&B CLI 또는 UI에서 job 을 실행할 때 사용자는 오직 네 가지 `trainer` 파라미터만 오버라이드 할 수 있습니다.

### run config 입력값 액세스하기

run config 입력값으로 실행된 job 은 `Run.config` 를 통해 이 값에 접근할 수 있습니다. job 코드 내의 `wandb.init` 로 반환된 `Run` 오브젝트에는 입력값이 자동으로 설정됩니다. 아래와 같이 활용하세요.
```python
from wandb.sdk import launch

run_config_overrides = launch.load_wandb_config()
```
이렇게 하면 job 코드 어디에서든 run config 입력값을 불러올 수 있습니다.

## 파일 재구성하기

Launch SDK 는 또한 job 코드 내 config 파일에 저장된 입력값을 관리하는 방법도 제공합니다. 이는 [torchtune](https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama3/8B_lora.yaml) 예시나 [Axolotl config](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/llama-3/qlora-fsdp-70b.yaml) 처럼 많은 딥러닝 및 대형 언어 모델 유스 케이스에서 흔히 사용됩니다.

{{% alert %}}
[Launch 에서의 Sweeps]({{< relref path="../sweeps-on-launch.md" lang="ko" >}}) 는 config 파일 입력값을 sweep 파라미터로 사용할 수 없습니다. sweep 파라미터는 반드시 `Run.config` 오브젝트를 통해 제어해야 합니다.
{{% /alert %}}

`launch.manage_config_file` 함수는 config 파일을 Launch job 의 입력값으로 추가할 때 사용할 수 있습니다. 이를 통해 job 을 실행할 때 해당 config 파일 내 값들을 수정할 수 있습니다.

이 함수를 사용할 경우, 기본적으로 run config 입력값은 캡처되지 않습니다. `launch.manage_wandb_config` 를 호출하면 이 동작이 변경됩니다.

아래 예시를 참고하세요:

```python
import yaml
import wandb
from wandb.sdk import launch

# launch sdk 사용을 위해 필수
wandb.require("core")

launch.manage_config_file("config.yaml")

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

with wandb.init(config=config):
    # 기타 코드 등등
    pass
```

이 코드가 아래와 같은 `config.yaml` 파일과 함께 실행된다면:

```yaml
learning_rate: 0.01
batch_size: 32
model: resnet
dataset: cifar10
```

`launch.manage_config_file` 호출은 `config.yaml` 파일을 job 입력값으로 추가하여, W&B CLI 또는 UI에서 job 실행 시 해당 파일을 수정할 수 있게 합니다.

`include` 및 `exclude` 키워드 인수는 config 파일에서 허용될 입력값 키를 `launch.manage_wandb_config` 와 동일하게 필터링하는 데 사용할 수 있습니다.

### config 파일 입력값 액세스하기

Launch 가 생성한 run 에서 `launch.manage_config_file` 이 호출되면, `launch` 가 입력값을 반영해 config 파일 내용을 패치합니다. 패치된 config 파일은 job 환경에서 사용 가능합니다.

{{% alert color="secondary" %}}
입력값을 반영하려면 job 코드에서 config 파일을 읽기 전에 `launch.manage_config_file` 을 호출하세요.
{{% /alert %}}

### job 의 launch drawer UI 커스터마이즈하기

job 입력값 스키마(schema)를 정의하면, job 을 실행할 때 커스텀 UI 를 생성할 수 있습니다. 스키마를 정의하려면, `launch.manage_wandb_config` 또는 `launch.manage_config_file` 호출 시 포함하세요. 스키마는 [JSON Schema](https://json-schema.org/understanding-json-schema/reference) 형태의 파이썬 딕셔너리거나, Pydantic 모델 클래스일 수 있습니다.

{{% alert color="secondary" %}}
job 입력값 스키마는 입력값 검증 용도가 아닙니다. launch drawer 의 UI 정의에만 사용됩니다.
{{% /alert %}}

{{< tabpane text=true >}}
{{% tab "JSON schema" %}}
다음 예시는 아래와 같은 속성을 가지는 스키마입니다.

- `seed`: 정수형
- `trainer`: 일부 키가 지정된 딕셔너리
  - `trainer.learning_rate`: 0보다 커야 하는 float
  - `trainer.batch_size`: 16, 64, 256 중 하나인 int
  - `trainer.dataset`: `cifar10` 또는 `cifar100` 이어야 하는 문자열

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
                    "description": "배치당 샘플 개수",
                    "enum": [16, 64, 256]
                },
                "dataset": {
                    "type": "string",
                    "description": "사용할 데이터셋 이름",
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

일반적으로 지원되는 JSON Schema 속성은 다음과 같습니다.

| 속성(Attribute) | 필수(Required) |  비고(Notes) |
| --- | --- | --- |
| `type` | 예 | `number`, `integer`, `string`, `object` 중 하나여야 합니다 |
| `title` | 아니오 | 속성의 표시 이름을 지정 |
| `description` | 아니오 | 속성 설명(도움말) |
| `enum` | 아니오 | 입력란을 드롭다운으로 만듦 |
| `minimum` | 아니오 | `type` 이 `number` 또는 `integer` 일 때 사용 |
| `maximum` | 아니오 | `type` 이 `number` 또는 `integer` 일 때 사용 |
| `exclusiveMinimum` | 아니오 | `type` 이 `number` 또는 `integer` 일 때 사용 |
| `exclusiveMaximum` | 아니오 | `type` 이 `number` 또는 `integer` 일 때 사용 |
| `properties` | 아니오 | `type` 이 `object` 일 때, 중첩 설정 정의 용도 |
{{% /tab %}}
{{% tab "Pydantic model" %}}
다음 예시는 아래와 같은 속성을 가지는 스키마입니다.

- `seed`: 정수형
- `trainer`: 하위 속성들이 명시된 스키마
  - `trainer.learning_rate`: 0보다 커야 하는 float
  - `trainer.batch_size`: 1 이상 256 이하인 int
  - `trainer.dataset`: `cifar10` 또는 `cifar100` 이어야 하는 문자열

```python
class DatasetEnum(str, Enum):
    cifar10 = "cifar10"
    cifar100 = "cifar100"

class Trainer(BaseModel):
    learning_rate: float = Field(gt=0, description="모델의 학습률")
    batch_size: int = Field(ge=1, le=256, description="배치당 샘플 개수")
    dataset: DatasetEnum = Field(title="데이터셋", description="사용할 데이터셋 이름")

class Schema(BaseModel):
    seed: int
    trainer: Trainer

launch.manage_wandb_config(
    include=["seed", "trainer"],
    exclude=["trainer.private"],
    schema=Schema,
)
```

또한 클래스 인스턴스를 직접 사용할 수도 있습니다.

```python
t = Trainer(learning_rate=0.01, batch_size=32, dataset=DatasetEnum.cifar10)
s = Schema(seed=42, trainer=t)
launch.manage_wandb_config(
    include=["seed", "trainer"],
    exclude=["trainer.private"],
    input_schema=s,
)
```
{{% /tab %}}
{{< /tabpane >}}

job 입력 스키마를 추가하면 launch drawer 에 구조화된 폼(form)이 생성되어 job 실행이 한결 쉬워집니다.

{{< img src="/images/launch/schema_overrides.png" alt="Job input schema form" >}}
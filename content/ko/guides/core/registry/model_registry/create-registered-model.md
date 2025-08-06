---
title: Registered 모델 생성하기
description: 등록된 model 을 생성하여 모델링 작업에 사용할 모든 후보 model 들을 관리하세요.
menu:
  default:
    identifier: ko-guides-core-registry-model_registry-create-registered-model
    parent: model-registry
weight: 4
---

[registered model]({{< relref path="./model-management-concepts.md#registered-model" lang="ko" >}})을 생성하여 모델링 작업에 사용할 모든 후보 모델을 보관하세요. Registered model은 Model Registry에서 인터랙티브하게 생성할 수 있고, Python SDK를 이용해 프로그래밍적으로 생성할 수도 있습니다.

## 프로그래밍적으로 registered model 생성하기
W&B Python SDK를 이용해 모델을 프로그래밍적으로 등록하세요. 만약 해당 registered model이 존재하지 않는다면, W&B가 자동으로 registered model을 생성합니다.

`<>`로 둘러싸인 값은 여러분의 값으로 반드시 바꿔주세요:

```python
import wandb

run = wandb.init(entity="<entity>", project="<project>")
run.link_model(path="<path-to-model>", registered_model_name="<registered-model-name>")
run.finish()
```

`registered_model_name`에 입력한 이름이 [Model Registry App](https://wandb.ai/registry/model)에 표시됩니다.

## 인터랙티브하게 registered model 생성하기
[Model Registry App](https://wandb.ai/registry/model)에서 registered model을 직접 생성할 수 있습니다.

1. [Model Registry App](https://wandb.ai/registry/model)으로 이동합니다.
{{< img src="/images/models/create_registered_model_1.png" alt="Model Registry landing page" >}}
2. Model Registry 페이지의 우측 상단에 위치한 **New registered model** 버튼을 클릭하세요.
{{< img src="/images/models/create_registered_model_model_reg_app.png" alt="New registered model button" >}}
3. 나타나는 패널에서 **Owning Entity** 드롭다운에서 registered model이 속할 entity를 선택하세요.
{{< img src="/images/models/create_registered_model_3.png" alt="Model creation form" >}}
4. **Name** 필드에 모델의 이름을 입력하세요.
5. **Type** 드롭다운에서, 등록할 registered model과 연결할 artifacts의 타입을 선택하세요.
6. (선택 사항) **Description** 필드에 모델에 대한 설명을 추가하세요.
7. (선택 사항) **Tags** 필드에 하나 이상의 태그를 추가하세요.
8. **Register model** 버튼을 클릭하세요.


{{% alert %}}
수동으로 모델을 model registry에 연결하는 것은 일회성 모델 관리에 유용합니다. 하지만 대부분의 경우 [프로그램적으로 모델 버전을 model registry에 연결하는 것]({{< relref path="link-model-version#programmatically-link-a-model" lang="ko" >}})이 더 효과적입니다.

예를 들어, 야간작업(nightly job)이 있다고 가정해 봅시다. 매일 생성되는 모델을 수동으로 model registry와 연결하는 일은 번거로울 수 있습니다. 대신, 모델의 성능을 평가하고 더 좋아졌다면 해당 모델을 W&B Python SDK를 이용해 model registry에 자동으로 연결하는 스크립트를 작성할 수 있습니다.
{{% /alert %}}
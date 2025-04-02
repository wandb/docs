---
title: Create a registered model
description: 모델링 작업을 위한 모든 후보 모델을 담을 등록된 모델 을 만드세요.
menu:
  default:
    identifier: ko-guides-core-registry-model_registry-create-registered-model
    parent: model-registry
weight: 4
---

[등록된 모델]({{< relref path="./model-management-concepts.md#registered-model" lang="ko" >}})을 생성하여 모델링 작업을 위한 모든 후보 모델을 보관하세요. Model Registry 내에서 대화식으로 또는 Python SDK를 사용하여 프로그래밍 방식으로 등록된 모델을 생성할 수 있습니다.

## 프로그래밍 방식으로 등록된 모델 생성
W&B Python SDK로 모델을 프로그래밍 방식으로 등록하세요. 등록된 모델이 존재하지 않으면 W&B가 자동으로 등록된 모델을 생성합니다.

`<>`로 묶인 다른 값들을 사용자의 값으로 바꾸세요:

```python
import wandb

run = wandb.init(entity="<entity>", project="<project>")
run.link_model(path="<path-to-model>", registered_model_name="<registered-model-name>")
run.finish()
```

`registered_model_name`에 제공하는 이름은 [Model Registry App](https://wandb.ai/registry/model)에 나타나는 이름입니다.

## 대화식으로 등록된 모델 생성
[Model Registry App](https://wandb.ai/registry/model) 내에서 대화식으로 등록된 모델을 생성하세요.

1. [https://wandb.ai/registry/model](https://wandb.ai/registry/model)에서 Model Registry App으로 이동합니다.
{{< img src="/images/models/create_registered_model_1.png" alt="" >}}
2. Model Registry 페이지의 오른쪽 상단에 있는 **New registered model** 버튼을 클릭합니다.
{{< img src="/images/models/create_registered_model_model_reg_app.png" alt="" >}}
3. 나타나는 패널에서 등록된 모델이 속할 엔터티를 **Owning Entity** 드롭다운에서 선택합니다.
{{< img src="/images/models/create_registered_model_3.png" alt="" >}}
4. **Name** 필드에 모델 이름을 입력합니다.
5. **Type** 드롭다운에서 등록된 모델에 연결할 아티팩트의 유형을 선택합니다.
6. (선택 사항) **Description** 필드에 모델에 대한 설명을 추가합니다.
7. (선택 사항) **Tags** 필드 내에서 하나 이상의 태그를 추가합니다.
8. **Register model**을 클릭합니다.

{{% alert %}}
모델을 Model Registry에 수동으로 연결하는 것은 일회성 모델에 유용합니다. 그러나 [모델 버전을 Model Registry에 프로그래밍 방식으로 연결]({{< relref path="link-model-version#programmatically-link-a-model" lang="ko" >}})하는 것이 유용한 경우가 많습니다.

예를 들어 야간 작업이 있다고 가정해 봅시다. 매일 밤 생성된 모델을 수동으로 연결하는 것은 지루합니다. 대신 모델을 평가하고 모델 성능이 향상되면 W&B Python SDK를 사용하여 해당 모델을 Model Registry에 연결하는 스크립트를 만들 수 있습니다.
{{% /alert %}}

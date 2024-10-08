---
title: Create a registered model
description: 등록된 모델을 생성하여 모델링 작업의 모든 후보 모델을 보관하세요.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

[registered model](./model-management-concepts.md#registered-model)을 만들어 여러분의 모델링 작업에 필요한 모든 후보 모델을 보관하세요. Model Registry 내에서 인터랙티브하게 또는 Python SDK를 사용해 프로그래밍적으로 registered model을 생성할 수 있습니다.

## 프로그램적으로 registered model 생성하기
W&B Python SDK를 사용하여 프로그램적으로 모델을 등록하세요. registered model이 존재하지 않을 경우, W&B는 자동으로 registered model을 생성합니다.

`<>`로 묶인 다른 값들은 여러분의 값으로 대체하십시오:

```python
import wandb

run = wandb.init(entity="<entity>", project="<project>")
run.link_model(path="<path-to-model>", registered_model_name="<registered-model-name>")
run.finish()
```

`registered_model_name`에 제공한 이름은 [Model Registry App](https://wandb.ai/registry/model)에 나타나는 이름입니다.

## 인터랙티브하게 registered model 생성하기
[Model Registry App](https://wandb.ai/registry/model) 내에서 인터랙티브하게 registered model을 생성하세요.

1. [https://wandb.ai/registry/model](https://wandb.ai/registry/model)에서 Model Registry App으로 이동하세요.
![](/images/models/create_registered_model_1.png)
2. Model Registry 페이지의 오른쪽 상단에 위치한 **New registered model** 버튼을 클릭하세요.
![](/images/models/create_registered_model_model_reg_app.png)
3. 나타나는 패널에서, **Owning Entity** 드롭다운에서 registered model이 속할 엔티티를 선택하세요.
![](/images/models/create_registered_model_3.png)
4. **Name** 필드에서 모델의 이름을 입력하세요. 
5. **Type** 드롭다운에서 registered model과 연결할 Artifacts의 타입을 선택하세요.
6. (선택 사항) **Description** 필드에 모델에 대한 설명을 추가하세요. 
7. (선택 사항) **Tags** 필드에 하나 이상의 태그를 추가하세요. 
8. **Register model**을 클릭하세요.

:::tip
모델 레지스트리에 모델을 수동으로 연결하는 것은 한 번 사용되는 모델에 유용합니다. 그러나, [프로그램적으로 모델 버전을 모델 레지스트리에 연결](link-model-version#programmatically-link-a-model)하는 것이 유용한 경우가 많습니다.

예를 들어, 매일 밤 수행되는 작업이 있다고 가정해 봅시다. 매일 밤 생성되는 모델을 수동으로 연결하는 것은 번거로울 수 있습니다. 대신, 모델을 평가하는 스크립트를 생성하고, 모델의 성능이 향상되면 W&B Python SDK를 사용하여 해당 모델을 모델 레지스트리에 연결할 수 있습니다.
:::
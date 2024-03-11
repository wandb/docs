---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 등록된 모델 생성하기

모델링 작업에 대한 후보 모델을 모두 보관할 [등록된 모델](./model-management-concepts.md#registered-model)을 생성하세요. 모델 레지스트리 내에서 대화형으로 또는 Python SDK를 사용하여 프로그래매틱하게 등록된 모델을 생성할 수 있습니다.

## 프로그래매틱하게 등록된 모델 생성하기
W&B Python SDK를 사용하여 프로그래매틱하게 모델을 등록하세요. 등록된 모델이 존재하지 않는 경우 W&B가 자동으로 등록된 모델을 생성합니다.

`<>` 안에 들어 있는 다른 값들을 꼭 자신의 것으로 교체하세요:

```python
import wandb

run = wandb.init(entity="<entity>", project="<project>")
run.link_model(path="<path-to-model>", registered_model_name="<registered-model-name>")
run.finish()
```

`registered_model_name`에 제공하는 이름은 [모델 레지스트리 앱](https://wandb.ai/registry/model)에 나타나는 이름입니다.

## 대화형으로 등록된 모델 생성하기
[모델 레지스트리 앱](https://wandb.ai/registry/model) 내에서 대화형으로 등록된 모델을 생성하세요.

1. [https://wandb.ai/registry/model](https://wandb.ai/registry/model)에 있는 모델 레지스트리 앱으로 이동하세요.
![](/images/models/create_registered_model_1.png)
2. 모델 레지스트리 페이지의 오른쪽 상단에 위치한 **새로운 등록된 모델** 버튼을 클릭하세요.
![](/images/models/create_registered_model_model_reg_app.png)
3. 나타나는 패널에서 **소유 엔티티** 드롭다운에서 등록된 모델이 속하길 원하는 엔티티를 선택하세요.
![](/images/models/create_registered_model_3.png)
4. **이름** 필드에 모델의 이름을 제공하세요.
5. **유형** 드롭다운에서 등록된 모델에 연결할 아티팩트 유형을 선택하세요.
6. (선택사항) **설명** 필드에 모델에 대한 설명을 추가하세요.
7. (선택사항) **태그** 필드 내에서 하나 이상의 태그를 추가하세요.
8. **모델 등록**을 클릭하세요.


:::tip
모델을 모델 레지스트리에 수동으로 연결하는 것은 일회성 모델에 유용합니다. 그러나, [모델 버전을 모델 레지스트리에 프로그래매틱하게 연결하는 것](#programmatically-link-a-model)이 종종 유용합니다.

예를 들어, 매일 밤 작업이 있다고 가정해 보세요. 매일 밤 생성되는 모델을 수동으로 연결하는 것은 번거로울 수 있습니다. 대신, 모델을 평가하고 모델의 성능이 향상되면 해당 모델을 W&B Python SDK를 사용하여 모델 레지스트리에 연결하는 스크립트를 생성할 수 있습니다.
:::
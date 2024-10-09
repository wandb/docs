---
title: Link a model version
description: 모델 버전을 Registered Models에 W&B 앱 또는 Python SDK를 사용하여 연결하세요.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

모델 버전을 등록된 모델에 W&B 앱 또는 Python SDK를 사용하여 프로그래매틱하게 연결하세요.

## 프로그래매틱하게 모델 연결하기

[`link_model`](../../ref/python/run.md#link_model) 메소드를 사용하여 모델 파일을 W&B run에 프로그래매틱하게 로그하고 W&B 모델 레지스트리와 연결하세요.

`<>`로 묶인 값들을 본인의 값으로 바꾸세요:

```python
import wandb

run = wandb.init(entity="<entity>", project="<project>")
run.link_model(path="<path-to-model>", registered_model_name="<registered-model-name>")
run.finish()
```

W&B는 `registered-model-name` 파라미터에 대해 지정한 이름이 이미 존재하지 않으면 등록된 모델을 생성합니다.

예를 들어, "Fine-Tuned-Review-Autocompletion"이라는 이름의 기존 등록된 모델(`registered-model-name="Fine-Tuned-Review-Autocompletion"`)이 있다고 가정해 봅시다. 그리고 몇몇 모델 버전이 여기에 연결되어 있다고 생각해 보세요: `v0`, `v1`, `v2`. 새 모델을 프로그래매틱하게 연결하고 동일한 등록된 모델 이름을 사용하면 (`registered-model-name="Fine-Tuned-Review-Autocompletion"`), W&B는 이 모델을 기존 등록된 모델에 연결하고 모델 버전 `v3`을 할당합니다. 이 이름의 등록된 모델이 없으면 새로운 등록 모델이 생성되고 이 모델은 모델 버전 `v0`을 갖게 됩니다.

["Fine-Tuned-Review-Autocompletion" 등록된 모델의 예를 여기에서 볼 수 있습니다](https://wandb.ai/reviewco/registry/model?selectionPath=reviewco%2Fmodel-registry%2FFinetuned-Review-Autocompletion&view=all-models).

## 인터랙티브하게 모델 연결하기

모델 레지스트리 또는 아티팩트 브라우저와 모델을 인터랙티브하게 연결하세요.

<Tabs
  defaultValue="model_ui"
  values={[
    {label: 'Model Registry', value: 'model_ui'},
    {label: 'Artifact browser', value: 'artifacts_ui'},
  ]}>
  <TabItem value="model_ui">

1. [https://wandb.ai/registry/model](https://wandb.ai/registry/model)에서 모델 레지스트리 앱으로 이동합니다.
2. 마우스를 등록된 모델 이름 옆에 두고 새 모델을 연결하려 합니다.
3. **View details** 옆의 세 개의 점 아이콘을 선택합니다.
4. 드롭다운에서 **Link new version**을 선택합니다.
5. **Project** 드롭다운에서 모델을 포함하는 프로젝트 이름을 선택합니다.
6. **Model Artifact** 드롭다운에서 모델 아티팩트 이름을 선택합니다.
7. **Version** 드롭다운에서 등록된 모델에 연결할 모델 버전을 선택합니다.

![](/images/models/link_model_wmodel_reg.gif)

  </TabItem>
  <TabItem value="artifacts_ui">

1. W&B 앱에서 프로젝트의 아티팩트 브라우저로 이동하세요: `https://wandb.ai/<entity>/<project>/artifacts`
2. 왼쪽 사이드바에서 Artifacts 아이콘을 선택합니다.
3. 레지스트리에 연결할 모델 버전을 클릭하세요.
4. **Version overview** 섹션에서 **Link to registry** 버튼을 클릭합니다.
5. 화면 오른쪽에 나타나는 모달에서 **Select a register model** 메뉴 드롭다운에서 등록된 모델을 선택하세요.
6. **Next step**을 클릭하세요.
7. (선택 사항) **Aliases** 드롭다운에서 에일리어스를 선택하세요.
8. **Link to registry**를 클릭합니다.

![](/images/models/manual_linking.gif)

  </TabItem>
</Tabs>

## 연결된 모델의 소스 보기

연결된 모델의 소스를 보는 방법은 두 가지가 있습니다: 모델이 로그된 프로젝트 내의 아티팩트 브라우저와 W&B 모델 레지스트리.

모델 레지스트리의 특정 모델 버전을 소스 모델 아티팩트에 연결하는 포인터가 있습니다(모델이 로그된 프로젝트 내에 위치). 소스 모델 아티팩트는 모델 레지스트리에 대한 포인터도 가지고 있습니다.

<Tabs
  defaultValue="registry"
  values={[
    {label: 'Model Registry', value: 'registry'},
    {label: 'Artifact browser', value: 'browser'},
  ]}>
  <TabItem value="registry">

1. [https://wandb.ai/registry/model](https://wandb.ai/registry/model)에서 모델 레지스트리로 이동합니다.
![](/images/models/create_registered_model_1.png)
2. 등록된 모델 이름 옆에 있는 **View details**를 선택합니다.
3. **Versions** 섹션에서 조사하려는 모델 버전 옆의 **View**를 선택합니다.
4. 오른쪽 패널 내의 **Version** 탭을 클릭합니다.
5. **Version overview** 섹션에는 **Source Version** 필드를 포함하는 행이 있습니다. **Source Version** 필드는 모델의 이름과 모델의 버전을 모두 보여줍니다.

예를 들어, 다음 이미지는 `mnist_model`이라는 `v0` 모델 버전을 보여주며 (**Source version** 필드 `mnist_model:v0` 참조), `MNIST-dev`라는 등록된 모델에 연결되어 있습니다.

![](/images/models/view_linked_model_registry.png)

  </TabItem>
  <TabItem value="browser">

1. W&B 앱에서 프로젝트의 아티팩트 브라우저로 이동하세요: `https://wandb.ai/<entity>/<project>/artifacts`
2. 왼쪽 사이드바에서 Artifacts 아이콘을 선택합니다.
3. Artifacts 패널에서 **model** 드롭다운 메뉴를 확장하세요.
4. 모델 레지스트리에 연결된 모델의 이름과 버전을 선택합니다.
5. 오른쪽 패널 내의 **Version** 탭을 클릭합니다.
6. **Version overview** 섹션에는 **Linked To** 필드를 포함하는 행이 있습니다. **Linked To** 필드는 등록된 모델의 이름과 그것의 버전을 보여줍니다(`registered-model-name:version`).

예를 들어, 다음 이미지에는 `MNIST-dev`라는 등록된 모델이 있습니다(참조 **Linked To** 필드). `v0` 버전을 가진 `mnist_model`이라는 모델 버전이 `MNIST-dev` 등록 모델을 가리킵니다.

![](/images/models/view_linked_model_artifacts_browser.png)

  </TabItem>
</Tabs>
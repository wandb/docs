---
title: 모델 버전 연결
description: W&B 앱이나 Python SDK를 사용하여 모델 버전을 Registered Model 에 연결할 수 있습니다.
menu:
  default:
    identifier: ko-guides-core-registry-model_registry-link-model-version
    parent: model-registry
weight: 5
---

모델 버전을 Registered Model 에 연결하려면 W&B App 또는 Python SDK를 통한 프로그래밍 방식 중 하나를 선택하세요.

## 프로그래밍 방식으로 모델 연결하기

[`link_model`]({{< relref path="/ref/python/sdk/classes/run.md#link_model" lang="ko" >}}) 메소드를 사용하면 모델 파일을 W&B run 에 로그하고 [W&B Model Registry]({{< relref path="./" lang="ko" >}})에 연결할 수 있습니다.

`<>`로 감싸진 값을 자신의 값으로 바꿔주세요:

```python
import wandb

run = wandb.init(entity="<entity>", project="<project>")
run.link_model(path="<path-to-model>", registered_model_name="<registered-model-name>")
run.finish()
```

`registered-model-name` 파라미터에 지정한 이름의 registered model 이 없으면, W&B가 새롭게 registered model 을 생성해 줍니다.

예를 들어, Model Registry 에 이미 "Fine-Tuned-Review-Autocompletion"(`registered-model-name="Fine-Tuned-Review-Autocompletion"`)이라는 registered model 이 존재하고 여기에 여러 모델 버전(`v0`, `v1`, `v2`)이 연결되어 있다고 가정해봅시다. 새 모델을 프로그래밍적으로 연결할 때 동일한 registered-model-name(`registered-model-name="Fine-Tuned-Review-Autocompletion"`)을 사용하면, W&B는 이 모델을 기존 registered model 에 연결하고 새로운 모델 버전 `v3`을 할당합니다. 만약 해당 이름의 registered model 이 존재하지 않으면 새로운 registered model 을 만들고 그 버전은 `v0`이 됩니다.

예시로 ["Fine-Tuned-Review-Autocompletion" registered model](https://wandb.ai/reviewco/registry/model?selectionPath=reviewco%2Fmodel-registry%2FFinetuned-Review-Autocompletion&view=all-models) 을 참고해보세요.

## 인터랙티브하게 모델 연결하기
Model Registry 또는 Artifact 브라우저를 통해 인터랙티브하게 모델을 연결할 수도 있습니다.

{{< tabpane text=true >}}
  {{% tab header="Model Registry" %}}
1. [Model Registry App](https://wandb.ai/registry/model)으로 이동하세요.
2. 새 모델을 연결하고 싶은 registered model 이름 옆으로 마우스를 가져가세요.
3. **View details** 옆에 있는 점 세 개(미트볼 아이콘)를 클릭하세요.
4. 드롭다운 메뉴에서 **Link new version**을 선택하세요.
5. **Project** 드롭다운에서 모델이 포함된 프로젝트 이름을 선택하세요.
6. **Model Artifact** 드롭다운에서 모델 artifact 이름을 선택하세요.
7. **Version** 드롭다운에서 registered model 에 연결하고 싶은 모델 버전을 선택하세요.

{{< img src="/images/models/link_model_wmodel_reg.gif" alt="모델 버전을 registry에 연결" >}}
  {{% /tab %}}
  {{% tab header="Artifact browser" %}}
1. W&B App에서 자신의 프로젝트 artifact 브라우저로 이동하세요: `https://wandb.ai/<entity>/<project>/artifacts`
2. 왼쪽 사이드바에서 Artifacts 아이콘을 클릭하세요.
3. Registry에 연결할 모델 버전을 클릭하세요.
4. **Version overview** 섹션에서 **Link to registry** 버튼을 클릭하세요.
5. 화면 오른쪽에 뜨는 모달에서 **Select a register model** 메뉴 드롭다운에서 연결할 registered model 을 선택하세요.
6. **Next step**을 클릭하세요.
7. (선택 사항) **Aliases** 드롭다운에서 에일리어스를 선택할 수 있습니다.
8. **Link to registry**를 클릭하세요.

{{< img src="/images/models/manual_linking.gif" alt="수동 모델 연결" >}}  
  {{% /tab %}}
{{< /tabpane >}}



## 연결된 모델의 소스 확인하기

연결된 모델의 소스를 확인하는 방법은 두 가지가 있습니다: 해당 모델이 로그된 프로젝트의 artifact 브라우저, 그리고 W&B Model Registry 입니다.

포인터로 모델 registry의 특정 모델 버전과 소스 모델 아티팩트(해당 모델이 업로드된 프로젝트 내 위치)가 연결됩니다. 소스 모델 아티팩트에도 model registry로의 포인터가 존재합니다.

{{< tabpane text=true >}}
  {{% tab header="Model Registry" %}}
1. [Model Registry App](https://wandb.ai/registry/model)으로 이동하세요.
{{< img src="/images/models/create_registered_model_1.png" alt="Registered model 생성" >}}
2. registered model 이름 옆의 **View details**를 선택하세요.
3. **Versions** 섹션에서 확인하고 싶은 모델 버전 옆의 **View**를 선택하세요.
4. 우측 패널의 **Version** 탭을 클릭하세요.
5. **Version overview** 섹션의 행에서 **Source Version** 필드를 찾을 수 있습니다. 이 필드는 모델 이름과 해당 모델의 버전을 모두 보여줍니다.

예를 들어, 아래 이미지는 `mnist_model`이라는 이름의 `v0` 모델 버전(**Source version** 필드에서 `mnist_model:v0` 확인)이 `MNIST-dev`라는 registered model 에 연결된 모습입니다.

{{< img src="/images/models/view_linked_model_registry.png" alt="Registry에 연결된 모델" >}}  
  {{% /tab %}}
  {{% tab header="Artifact browser" %}}
1. W&B App에서 자신의 프로젝트 artifact 브라우저로 이동하세요: `https://wandb.ai/<entity>/<project>/artifacts`
2. 왼쪽 사이드바에서 Artifacts 아이콘을 클릭하세요.
3. Artifacts 패널에서 **model** 드롭다운 메뉴를 펼치세요.
4. Model registry에 연결된 모델의 이름과 버전을 선택하세요.
5. 우측 패널에서 **Version** 탭을 클릭하세요.
6. **Version overview** 섹션의 행에서 **Linked To** 필드를 확인할 수 있습니다. 이 필드는 registered model 이름과 할당된 버전(`registered-model-name:version`)을 보여줍니다.

예를 들어, 아래 이미지처럼 `MNIST-dev`라는 registered model 이 존재하고(**Linked To** 필드 참고), `mnist_model`이라는 모델 버전 `v0`(`mnist_model:v0`)이 이 registered model 을 가리키고 있음을 확인할 수 있습니다.

{{< img src="/images/models/view_linked_model_artifacts_browser.png" alt="Model artifacts 브라우저" >}}  
  {{% /tab %}}
{{< /tabpane >}}
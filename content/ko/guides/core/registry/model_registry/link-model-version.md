---
title: Link a model version
description: W&B 앱 또는 Python SDK를 사용하여 모델 버전을 등록된 모델에 연결합니다.
menu:
  default:
    identifier: ko-guides-core-registry-model_registry-link-model-version
    parent: model-registry
weight: 5
---

W&B 앱 또는 Python SDK를 사용하여 모델 버전을 등록된 모델에 연결합니다.

## 프로그램으로 모델 연결하기

[`link_model`]({{< relref path="/ref/python/run.md#link_model" lang="ko" >}}) 메소드를 사용하여 프로그램 방식으로 모델 파일을 W&B run에 로그하고 [W&B Model Registry]({{< relref path="./" lang="ko" >}})에 연결합니다.

`<>`로 묶인 다른 값들을 사용자 정의 값으로 바꾸십시오:

```python
import wandb

run = wandb.init(entity="<entity>", project="<project>")
run.link_model(path="<path-to-model>", registered_model_name="<registered-model-name>")
run.finish()
```

`registered-model-name` 파라미터에 지정한 이름이 아직 존재하지 않는 경우, W&B가 등록된 모델을 생성합니다.

예를 들어, Model Registry에 "Fine-Tuned-Review-Autocompletion"(`registered-model-name="Fine-Tuned-Review-Autocompletion"`)이라는 등록된 모델이 이미 있다고 가정합니다. 그리고 몇몇 모델 버전이 연결되어 있다고 가정합니다: `v0`, `v1`, `v2`. 새로운 모델을 프로그램 방식으로 연결하고 동일한 등록된 모델 이름(`registered-model-name="Fine-Tuned-Review-Autocompletion"`)을 사용하면, W&B는 이 모델을 기존 등록된 모델에 연결하고 모델 버전 `v3`을 할당합니다. 이 이름으로 등록된 모델이 없으면 새로운 등록된 모델이 생성되고 모델 버전 `v0`을 갖게 됩니다.

["Fine-Tuned-Review-Autocompletion" 등록된 모델 예시](https://wandb.ai/reviewco/registry/model?selectionPath=reviewco%2Fmodel-registry%2FFinetuned-Review-Autocompletion&view=all-models)를 참조하십시오.

## 대화형으로 모델 연결하기
Model Registry 또는 Artifact browser를 사용하여 대화형으로 모델을 연결합니다.

{{< tabpane text=true >}}
  {{% tab header="Model Registry" %}}
1. Model Registry 앱([https://wandb.ai/registry/model](https://wandb.ai/registry/model))으로 이동합니다.
2. 새 모델을 연결하려는 등록된 모델 이름 옆에 마우스를 올려 놓습니다.
3. **View details** 옆에 있는 미트볼 메뉴 아이콘(가로 점 3개)을 선택합니다.
4. 드롭다운에서 **Link new version**을 선택합니다.
5. **Project** 드롭다운에서 모델이 포함된 프로젝트 이름을 선택합니다.
6. **Model Artifact** 드롭다운에서 모델 아티팩트 이름을 선택합니다.
7. **Version** 드롭다운에서 등록된 모델에 연결하려는 모델 버전을 선택합니다.

{{< img src="/images/models/link_model_wmodel_reg.gif" alt="" >}}
  {{% /tab %}}
  {{% tab header="Artifact browser" %}}
1. W&B 앱의 프로젝트 아티팩트 브라우저(`https://wandb.ai/<entity>/<project>/artifacts`)로 이동합니다.
2. 왼쪽 사이드바에서 Artifacts 아이콘을 선택합니다.
3. 레지스트리에 연결하려는 모델 버전을 클릭합니다.
4. **Version overview** 섹션 내에서 **Link to registry** 버튼을 클릭합니다.
5. 화면 오른쪽에 나타나는 모달에서 **Select a register model** 메뉴 드롭다운에서 등록된 모델을 선택합니다.
6. **Next step**을 클릭합니다.
7. (선택 사항) **Aliases** 드롭다운에서 에일리어스를 선택합니다.
8. **Link to registry**를 클릭합니다.

{{< img src="/images/models/manual_linking.gif" alt="" >}}
  {{% /tab %}}
{{< /tabpane >}}

## 연결된 모델의 소스 보기

연결된 모델의 소스를 보는 방법은 두 가지가 있습니다: 모델이 로그된 프로젝트 내의 아티팩트 브라우저와 W&B Model Registry.

포인터는 모델 레지스트리의 특정 모델 버전을 소스 모델 아티팩트(모델이 로그된 프로젝트 내에 있음)에 연결합니다. 소스 모델 아티팩트에는 모델 레지스트리에 대한 포인터도 있습니다.

{{< tabpane text=true >}}
  {{% tab header="Model Registry" %}}
1. 모델 레지스트리([https://wandb.ai/registry/model](https://wandb.ai/registry/model))로 이동합니다.
{{< img src="/images/models/create_registered_model_1.png" alt="" >}}
2. 등록된 모델 이름 옆에 있는 **View details**를 선택합니다.
3. **Versions** 섹션 내에서 조사하려는 모델 버전 옆에 있는 **View**를 선택합니다.
4. 오른쪽 패널 내에서 **Version** 탭을 클릭합니다.
5. **Version overview** 섹션 내에 **Source Version** 필드가 포함된 행이 있습니다. **Source Version** 필드는 모델 이름과 모델 버전을 모두 보여줍니다.

예를 들어, 다음 이미지는 `MNIST-dev`라는 등록된 모델에 연결된 `v0` 모델 버전 `mnist_model`을 보여줍니다( **Source version** 필드 `mnist_model:v0` 참조).

{{< img src="/images/models/view_linked_model_registry.png" alt="" >}}
  {{% /tab %}}
  {{% tab header="Artifact browser" %}}
1. W&B 앱의 프로젝트 아티팩트 브라우저(`https://wandb.ai/<entity>/<project>/artifacts`)로 이동합니다.
2. 왼쪽 사이드바에서 Artifacts 아이콘을 선택합니다.
3. Artifacts 패널에서 **model** 드롭다운 메뉴를 확장합니다.
4. 모델 레지스트리에 연결된 모델의 이름과 버전을 선택합니다.
5. 오른쪽 패널 내에서 **Version** 탭을 클릭합니다.
6. **Version overview** 섹션 내에 **Linked To** 필드가 포함된 행이 있습니다. **Linked To** 필드는 등록된 모델 이름과 해당 버전(`registered-model-name:version`)을 모두 보여줍니다.

예를 들어, 다음 이미지에는 `MNIST-dev`라는 등록된 모델이 있습니다( **Linked To** 필드 참조). 버전 `v0`(`mnist_model:v0`)이 있는 `mnist_model`이라는 모델 버전은 `MNIST-dev` 등록된 모델을 가리킵니다.

{{< img src="/images/models/view_linked_model_artifacts_browser.png" alt="" >}}
  {{% /tab %}}
{{< /tabpane >}}

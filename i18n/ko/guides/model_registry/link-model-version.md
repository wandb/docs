---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 모델 버전 연결하기

W&B 앱이나 Python SDK를 통해 등록된 모델에 모델 버전을 연결하세요.

## 프로그래매틱하게 모델 연결하기

[`link_model`](../../ref/python/run.md#link_model) 메서드를 사용하여 모델 파일을 W&B 실행에 프로그래매틱하게 로그하고 [W&B 모델 레지스트리](./intro.md)에 연결하세요.

`<>`로 둘러싸인 값을 귀하의 것으로 교체하세요:

```python
import wandb

run = wandb.init(entity="<entity>", project="<project>")
run.link_model(path="<path-to-model>", registered_model_name="<registered-model-name>")
run.finish()
```

`registered-model-name` 파라미터에 지정한 이름이 이미 존재하지 않는 경우 W&B가 등록된 모델을 생성합니다.

예를 들어, 모델 레지스트리에 "Fine-Tuned-Review-Autocompletion"(`registered-model-name="Fine-Tuned-Review-Autocompletion"`)이라는 이름의 기존 등록된 모델이 있고, 이에 연결된 몇 가지 모델 버전이 있다고 가정해 보세요: `v0`, `v1`, `v2`. 만약 프로그래매틱하게 새 모델을 연결하고 같은 등록된 모델 이름(`registered-model-name="Fine-Tuned-Review-Autocompletion"`)을 사용한다면, W&B는 이 모델을 기존 등록된 모델에 연결하고 모델 버전 `v3`를 할당합니다. 이 이름을 가진 등록된 모델이 존재하지 않는 경우, 새로운 등록된 모델이 생성되며 모델 버전 `v0`을 가집니다.

["Fine-Tuned-Review-Autocompletion" 등록된 모델의 예시 여기에서 보기](https://wandb.ai/reviewco/registry/model?selectionPath=reviewco%2Fmodel-registry%2FFinetuned-Review-Autocompletion&view=all-models).

## 상호작용적으로 모델 연결하기
모델 레지스트리나 아티팩트 브라우저를 사용하여 상호작용적으로 모델을 연결하세요.

<Tabs
  defaultValue="model_ui"
  values={[
    {label: '모델 레지스트리', value: 'model_ui'},
    {label: '아티팩트 브라우저', value: 'artifacts_ui'},
  ]}>
  <TabItem value="model_ui">

1. [https://wandb.ai/registry/model](https://wandb.ai/registry/model)에서 모델 레지스트리 앱으로 이동하세요.
2. 새 모델을 연결하고자 하는 등록된 모델 이름 옆에서 마우스를 호버하세요.
3. **세부 정보 보기** 옆에 있는 미트볼 메뉴 아이콘(세 개의 수평 점)을 선택하세요.
4. 드롭다운에서 **새 버전 연결**을 선택하세요.
5. **프로젝트** 드롭다운에서 모델이 포함된 프로젝트의 이름을 선택하세요.
6. **모델 아티팩트** 드롭다운에서 모델 아티팩트의 이름을 선택하세요.
7. **버전** 드롭다운에서 등록된 모델에 연결하고자 하는 모델 버전을 선택하세요.

![](/images/models/link_model_wmodel_reg.gif)

  </TabItem>
  <TabItem value="artifacts_ui">

1. W&B 앱에서 귀하의 프로젝트의 아티팩트 브라우저로 이동하세요: `https://wandb.ai/<entity>/<project>/artifacts`
2. 왼쪽 사이드바에서 아티팩트 아이콘을 선택하세요.
3. 레지스트리에 연결하고자 하는 모델 버전을 클릭하세요.
4. **버전 개요** 섹션 내에서 **레지스트리에 연결** 버튼을 클릭하세요.
5. 화면 오른쪽에 나타나는 모달에서 **모델 등록 선택** 메뉴 드롭다운에서 등록된 모델을 선택하세요.
6. **다음 단계**를 클릭하세요.
7. (선택사항) **별칭** 드롭다운에서 별칭을 선택하세요.
8. **레지스트리에 연결**을 클릭하세요.

![](/images/models/manual_linking.gif)

  </TabItem>
</Tabs>

## 연결된 모델의 출처 보기

연결된 모델의 출처를 보는 두 가지 방법이 있습니다: 모델이 로그된 프로젝트 내의 아티팩트 브라우저와 W&B 모델 레지스트리입니다.

특정 모델 버전을 모델 레지스트리에 연결하는 포인터가 있으며, 출처 모델 아티팩트(모델이 로그된 프로젝트 내에 위치함)도 모델 레지스트리를 가리키는 포인터를 갖습니다.

<Tabs
  defaultValue="registry"
  values={[
    {label: '모델 레지스트리', value: 'registry'},
    {label: '아티팩트 브라우저', value: 'browser'},
  ]}>
  <TabItem value="registry">

1. [https://wandb.ai/registry/model](https://wandb.ai/registry/model)에서 귀하의 모델 레지스트리로 이동하세요.
![](/images/models/create_registered_model_1.png)
2. 등록된 모델의 이름 옆에 있는 **세부 정보 보기**를 선택하세요.
3. **버전** 섹션에서 조사하고자 하는 모델 버전 옆에 있는 **보기**를 선택하세요.
4. 오른쪽 패널 내의 **버전** 탭을 클릭하세요.
5. **버전 개요** 섹션에는 **출처 버전** 필드가 포함된 행이 있습니다. **출처 버전** 필드는 모델의 이름과 모델의 버전을 모두 보여줍니다.

예를 들어, 아래 이미지는 `MNIST-dev`라는 등록된 모델에 연결된 `mnist_model`이라고 불리는 `v0` 모델 버전을 보여줍니다( **출처 버전** 필드 `mnist_model:v0` 참조).

![](/images/models/view_linked_model_registry.png)

  </TabItem>
  <TabItem value="browser">

1. W&B 앱에서 `https://wandb.ai/<entity>/<project>/artifacts`로 귀하의 프로젝트의 아티팩트 브라우저로 이동하세요.
2. 왼쪽 사이드바에서 아티팩트 아이콘을 선택하세요.
3. 아티팩트 패널에서 **모델** 드롭다운 메뉴를 확장하세요.
4. 모델 레지스트리에 연결된 모델의 이름과 버전을 선택하세요.
5. 오른쪽 패널 내의 **버전** 탭을 클릭하세요.
6. **버전 개요** 섹션에는 **연결됨** 필드가 포함된 행이 있습니다. **연결됨** 필드는 등록된 모델의 이름과 그것이 가지고 있는 버전을 모두 보여줍니다(`registered-model-name:version`).

예를 들어, 아래 이미지에서는 `MNIST-dev`라는 등록된 모델을 볼 수 있습니다(**연결됨** 필드 참조). 모델 버전 `mnist_model`과 버전 `v0`(`mnist_model:v0`)가 `MNIST-dev` 등록된 모델을 가리킵니다.


![](/images/models/view_linked_model_artifacts_browser.png)


  </TabItem>
</Tabs>
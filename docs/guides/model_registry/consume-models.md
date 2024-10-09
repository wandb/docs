---
title: Download a model version
description: W&B Python SDK로 모델 다운로드하는 방법
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

W&B Python SDK를 사용하여 Model Registry에 연결한 모델 아티팩트를 다운로드하세요.

:::info
모델을 재구성하고 역직렬화하여 작업할 수 있는 형태로 변환하기 위해 추가적인 Python 함수와 API 호출을 제공해야 합니다.

W&B는 모델 카드를 사용하여 메모리에 모델을 로드하는 방법에 대한 정보를 문서화할 것을 권장합니다. 자세한 내용은 [기계학습 모델 문서화](./create-model-cards.md) 페이지를 참조하세요.
:::

`<>` 안의 값을 자신의 것으로 대체하세요:

```python
import wandb

# run 초기화
run = wandb.init(project="<project>", entity="<entity>")

# 모델 엑세스 및 다운로드. 다운로드된 아티팩트의 경로를 반환함
downloaded_model_path = run.use_model(name="<your-model-name>")
```

다음에 나열된 형식 중 하나를 사용하여 모델 버전을 참조하세요:

* `latest` - 가장 최근에 연결된 모델 버전을 지정하기 위해 `latest` 에일리어스를 사용하세요.
* `v#` - `v0`, `v1`, `v2` 등을 사용하여 Registered Model에서 특정 버전을 가져옵니다.
* `alias` - 커스텀 에일리어스를 지정하여 팀과 함께 모델 버전에 할당한 별칭을 사용하세요.

가능한 파라미터와 반환 타입에 대한 자세한 내용은 API Reference 가이드의 [`use_model`](../../ref/python/run.md#use_model)를 참조하세요.

<details>
<summary>예시: 로그된 모델 다운로드 및 사용하기</summary>

예를 들어, 다음의 코드조각에서는 사용자가 `use_model` API를 호출했습니다. 그들은 가져오고자 하는 모델 아티팩트의 이름을 지정했으며 버전/에일리어스도 제공했습니다. 그런 다음 API로부터 반환된 경로를 `downloaded_model_path` 변수에 저장했습니다.

```python
import wandb

entity = "luka"
project = "NLP_Experiments"
alias = "latest"  # 모델 버전에 대한 시멘틱 별칭 또는 식별자
model_artifact_name = "fine-tuned-model"

# run 초기화
run = wandb.init()
# 모델 엑세스 및 다운로드. 다운로드된 아티팩트의 경로를 반환함

downloaded_model_path = run.use_model(name=f"{entity/project/model_artifact_name}:{alias}")
```
</details>

:::caution 2024년 W&B Model Registry의 계획된 폐기
다음 탭에서는 곧 폐기될 Model Registry를 사용하여 모델 아티팩트를 처리하는 방법을 설명합니다.

W&B Registry를 사용하여 모델 아티팩트를 추적, 조직 및 소비할 수 있습니다. 자세한 사항은 [Registry 문서](../registry/intro.md)를 참조하세요.
:::

<Tabs
  defaultValue="cli"
  values={[
    {label: 'CLI', value: 'cli'},
    {label: 'W&B App', value: 'app'},
  ]}>
  <TabItem value="cli">

`<>` 안의 값을 자신의 것으로 대체하세요:
```python
import wandb
# run 초기화
run = wandb.init(project="<project>", entity="<entity>")
# 모델 엑세스 및 다운로드. 다운로드된 아티팩트의 경로를 반환함
downloaded_model_path = run.use_model(name="<your-model-name>")
```
다음에 나열된 형식 중 하나를 사용하여 모델 버전을 참조하세요:

* `latest` - 가장 최근에 연결된 모델 버전을 지정하기 위해 `latest` 에일리어스를 사용하세요.
* `v#` - `v0`, `v1`, `v2` 등을 사용하여 Registered Model에서 특정 버전을 가져옵니다.
* `alias` - 커스텀 에일리어스를 지정하여 팀과 함께 모델 버전에 할당한 별칭을 사용하세요.

가능한 파라미터와 반환 타입에 대한 자세한 내용은 API Reference 가이드의 [`use_model`](../../ref/python/run.md#use_model)를 참조하세요.

  </TabItem>
  <TabItem value="app">

1. [https://wandb.ai/registry/model](https://wandb.ai/registry/model)에서 Model Registry App으로 이동합니다.
2. 다운로드하려는 모델이 포함된 등록된 모델 이름 옆의 **View details**를 선택합니다.
3. Versions 섹션에서 다운로드하려는 모델 버전 옆의 View 버튼을 선택합니다.
4. **Files** 탭을 선택합니다.
5. 다운로드하려는 모델 파일 옆의 다운로드 버튼을 클릭합니다.
![](/images/models/download_model_ui.gif)

  </TabItem>
</Tabs>
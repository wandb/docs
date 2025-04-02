---
title: Download a model version
description: W&B Python SDK로 모델을 다운로드하는 방법
menu:
  default:
    identifier: ko-guides-core-registry-model_registry-consume-models
    parent: model-registry
weight: 8
---

W&B Python SDK를 사용하여 Model Registry에 연결한 모델 아티팩트를 다운로드합니다.

{{% alert %}}
모델을 재구성하고, 역직렬화하여 사용할 수 있는 형태로 만들려면 추가적인 Python 함수와 API 호출을 제공해야 합니다.

W&B에서는 모델을 메모리에 로드하는 방법에 대한 정보를 모델 카드를 통해 문서화할 것을 권장합니다. 자세한 내용은 [기계 학습 모델 문서화]({{< relref path="./create-model-cards.md" lang="ko" >}}) 페이지를 참조하세요.
{{% /alert %}}

`<>` 안의 값을 직접 변경하세요:

```python
import wandb

# run 초기화
run = wandb.init(project="<project>", entity="<entity>")

# 모델에 엑세스하고 다운로드합니다. 다운로드된 아티팩트의 경로를 반환합니다.
downloaded_model_path = run.use_model(name="<your-model-name>")
```

다음 형식 중 하나를 사용하여 모델 버전을 참조하세요:

* `latest` - 가장 최근에 연결된 모델 버전을 지정하려면 `latest` 에일리어스를 사용합니다.
* `v#` - Registered Model에서 특정 버전을 가져오려면 `v0`, `v1`, `v2` 등을 사용합니다.
* `alias` - 팀에서 모델 버전에 할당한 사용자 지정 에일리어스를 지정합니다.

가능한 파라미터 및 반환 유형에 대한 자세한 내용은 API Reference 가이드의 [`use_model`]({{< relref path="/ref/python/run.md#use_model" lang="ko" >}})을 참조하세요.

<details>
<summary>예시: 기록된 모델 다운로드 및 사용</summary>

예를 들어, 다음 코드 조각에서 사용자는 `use_model` API를 호출했습니다. 가져오려는 모델 아티팩트의 이름을 지정하고 버전/에일리어스도 제공했습니다. 그런 다음 API에서 반환된 경로를 `downloaded_model_path` 변수에 저장했습니다.

```python
import wandb

entity = "luka"
project = "NLP_Experiments"
alias = "latest"  # 모델 버전에 대한 시맨틱 닉네임 또는 식별자
model_artifact_name = "fine-tuned-model"

# run 초기화
run = wandb.init()
# 모델에 엑세스하고 다운로드합니다. 다운로드된 아티팩트의 경로를 반환합니다.

downloaded_model_path = run.use_model(name=f"{entity/project/model_artifact_name}:{alias}")
```
</details>


{{% alert title="2024년 W&B Model Registry 지원 중단 예정" %}}
다음 탭은 곧 지원이 중단될 Model Registry를 사용하여 모델 아티팩트를 사용하는 방법을 보여줍니다.

W&B Registry를 사용하여 모델 아티팩트를 추적, 구성 및 사용합니다. 자세한 내용은 [Registry 문서]({{< relref path="/guides/core/registry/" lang="ko" >}})를 참조하세요.
{{% /alert %}}

{{< tabpane text=true >}}
  {{% tab header="CLI" %}}
`<>` 안의 값을 직접 변경하세요:
```python
import wandb
# run 초기화
run = wandb.init(project="<project>", entity="<entity>")
# 모델에 엑세스하고 다운로드합니다. 다운로드된 아티팩트의 경로를 반환합니다.
downloaded_model_path = run.use_model(name="<your-model-name>")
```
다음 형식 중 하나를 사용하여 모델 버전을 참조하세요:

* `latest` - 가장 최근에 연결된 모델 버전을 지정하려면 `latest` 에일리어스를 사용합니다.
* `v#` - Registered Model에서 특정 버전을 가져오려면 `v0`, `v1`, `v2` 등을 사용합니다.
* `alias` - 팀에서 모델 버전에 할당한 사용자 지정 에일리어스를 지정합니다.

가능한 파라미터 및 반환 유형에 대한 자세한 내용은 API Reference 가이드의 [`use_model`]({{< relref path="/ref/python/run.md#use_model" lang="ko" >}})을 참조하세요.
  {{% /tab %}}
  {{% tab header="W&B App" %}}
1. [https://wandb.ai/registry/model](https://wandb.ai/registry/model)에서 Model Registry App으로 이동합니다.
2. 다운로드하려는 모델이 포함된 Registered Model 이름 옆에 있는 **세부 정보 보기**를 선택합니다.
3. 버전 섹션에서 다운로드하려는 모델 버전 옆에 있는 보기 버튼을 선택합니다.
4. **파일** 탭을 선택합니다.
5. 다운로드하려는 모델 파일 옆에 있는 다운로드 버튼을 클릭합니다.
{{< img src="/images/models/download_model_ui.gif" alt="" >}}
  {{% /tab %}}
{{< /tabpane >}}

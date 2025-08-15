---
title: 모델 버전 다운로드
description: W&B Python SDK로 모델 다운로드하는 방법
menu:
  default:
    identifier: ko-guides-core-registry-model_registry-consume-models
    parent: model-registry
weight: 8
---

W&B Python SDK를 사용하여 Model Registry에 연결된 모델 artifact를 다운로드할 수 있습니다.

{{% alert %}}
추가적인 Python 함수나 API 호출을 통해 모델을 재구성하고, 직렬 해제하여 사용할 수 있는 형태로 복원하는 것은 사용자의 책임입니다.

W&B는 model card를 통해 메모리로 모델을 로드하는 방법에 대한 정보를 문서화할 것을 권장합니다. 자세한 내용은 [기계학습 모델 문서화]({{< relref path="./create-model-cards.md" lang="ko" >}}) 페이지를 참고하세요.
{{% /alert %}}

`<>` 안의 값을 본인에 맞게 변경하세요:

```python
import wandb

# run을 초기화합니다
run = wandb.init(project="<project>", entity="<entity>")

# 모델에 엑세스하고 다운로드합니다. 다운로드된 artifact의 경로가 반환됩니다
downloaded_model_path = run.use_model(name="<your-model-name>")
```

모델 버전은 아래 형식 중 하나로 지정할 수 있습니다:

* `latest` - 모델 버전 중 가장 최근에 연결된 것을 가리키려면 `latest` 에일리어스를 사용하세요.
* `v#` - `v0`, `v1`, `v2` 등으로 Registered Model의 특정 버전을 불러올 수 있습니다.
* `alias` - 본인이나 팀이 해당 모델 버전에 지정한 커스텀 에일리어스를 사용할 수 있습니다.

가능한 파라미터와 반환 타입에 대한 자세한 내용은 API Reference 가이드의 [`use_model`]({{< relref path="/ref/python/sdk/classes/run.md#use_model" lang="ko" >}})를 참고하세요.

<details>
<summary>예시: 로그인된 모델 다운로드 및 사용</summary>

아래 코드조각에서는 사용자가 `use_model` API를 호출했습니다. 원하는 모델 artifact의 이름과 버전/에일리어스를 명시했고, API에서 반환된 경로를 `downloaded_model_path` 변수에 저장했습니다.

```python
import wandb

entity = "luka"
project = "NLP_Experiments"
alias = "latest"  # 모델 버전에 대한 명확한 별칭(식별자)
model_artifact_name = "fine-tuned-model"

# run을 초기화합니다
run = wandb.init()
# 모델에 엑세스하고 다운로드합니다. 다운로드된 artifact의 경로가 반환됩니다

downloaded_model_path = run.use_model(name=f"{entity/project/model_artifact_name}:{alias}")
```
</details>

{{% alert title="2024년에 예정된 W&B Model Registry 폐기 안내" %}}
아래 탭에서는 곧 폐기 예정인 Model Registry를 통해 모델 artifact를 활용하는 방법을 보여줍니다.

W&B Registry를 사용하여 model artifact를 관리, 추적, 소비하세요. 더 자세한 내용은 [Registry docs]({{< relref path="/guides/core/registry/" lang="ko" >}})를 참고하세요.
{{% /alert %}}

{{< tabpane text=true >}}
  {{% tab header="CLI" %}}
`<>` 안의 값을 본인에 맞게 변경하세요:
```python
import wandb
# run을 초기화합니다
run = wandb.init(project="<project>", entity="<entity>")
# 모델에 엑세스하고 다운로드합니다. 다운로드된 artifact의 경로가 반환됩니다
downloaded_model_path = run.use_model(name="<your-model-name>")
```
모델 버전은 아래 형식 중 하나로 지정할 수 있습니다:

* `latest` - 모델 버전 중 가장 최근에 연결된 것을 가리키려면 `latest` 에일리어스를 사용하세요.
* `v#` - `v0`, `v1`, `v2` 등으로 Registered Model의 특정 버전을 불러올 수 있습니다.
* `alias` - 본인이나 팀이 해당 모델 버전에 지정한 커스텀 에일리어스를 사용할 수 있습니다.

파라미터와 반환 타입에 대한 내용은 API Reference 가이드의 [`use_model`]({{< relref path="/ref/python/sdk/classes/run.md#use_model" lang="ko" >}})를 참고하세요.  
  {{% /tab %}}
  {{% tab header="W&B App" %}}
1. [Model Registry App](https://wandb.ai/registry/model)으로 이동합니다.
2. 다운로드하고 싶은 모델을 포함한 등록된 모델 이름 옆의 **View details**를 선택합니다.
3. Versions 섹션에서 다운로드받고 싶은 모델 버전 옆의 View 버튼을 선택합니다.
4. **Files** 탭을 선택합니다.
5. 다운로드하고 싶은 모델 파일 옆의 다운로드 버튼을 클릭하세요.
{{< img src="/images/models/download_model_ui.gif" alt="UI에서 모델 다운로드" >}}  
  {{% /tab %}}
{{< /tabpane >}}
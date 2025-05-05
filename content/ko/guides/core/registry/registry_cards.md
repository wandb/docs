---
title: Annotate collections
menu:
  default:
    identifier: ko-guides-core-registry-registry_cards
    parent: registry
weight: 8
---

컬렉션에 사람이 읽기 쉬운 텍스트를 추가하여 사용자가 컬렉션의 목적과 컬렉션에 포함된 아티팩트를 이해하는 데 도움이 되도록 하세요.

컬렉션에 따라 트레이닝 데이터, 모델 아키텍처, 작업, 라이선스, 참조 및 배포에 대한 정보를 포함할 수 있습니다. 다음은 컬렉션에 문서화할 가치가 있는 몇 가지 주제를 나열한 것입니다.

W&B는 최소한 다음 세부 정보를 포함할 것을 권장합니다.
* **요약**: 컬렉션의 목적. 기계 학습 실험에 사용된 기계 학습 프레임워크.
* **라이선스**: 기계 학습 모델 사용과 관련된 법적 조건 및 권한. 모델 사용자가 모델을 활용할 수 있는 법적 프레임워크를 이해하는 데 도움이 됩니다. 일반적인 라이선스에는 Apache 2.0, MIT 및 GPL이 있습니다.
* **참조**: 관련 연구 논문, 데이터셋 또는 외부 리소스에 대한 인용 또는 참조.

컬렉션에 트레이닝 데이터가 포함된 경우 다음 추가 세부 정보를 포함하는 것을 고려하십시오.
* **트레이닝 데이터**: 사용된 트레이닝 데이터에 대해 설명합니다.
* **처리**: 트레이닝 데이터 세트에서 수행된 처리.
* **데이터 저장소**: 해당 데이터가 저장된 위치 및 엑세스 방법.

컬렉션에 기계 학습 모델이 포함된 경우 다음 추가 세부 정보를 포함하는 것을 고려하십시오.
* **아키텍처**: 모델 아키텍처, 레이어 및 특정 설계 선택에 대한 정보.
* **작업**: 컬렉션 모델이 수행하도록 설계된 특정 유형의 작업 또는 문제. 모델의 의도된 기능을 분류한 것입니다.
* **모델 역직렬화**: 팀의 누군가가 모델을 메모리에 로드할 수 있는 방법에 대한 정보를 제공합니다.
* **작업**: 기계 학습 모델이 수행하도록 설계된 특정 유형의 작업 또는 문제입니다. 모델의 의도된 기능을 분류한 것입니다.
* **배포**: 모델이 배포되는 방식 및 위치에 대한 세부 정보와 모델을 워크플로우 오케스트레이션 플랫폼과 같은 다른 엔터프라이즈 시스템에 통합하는 방법에 대한 지침.

## 컬렉션에 대한 설명 추가

W&B Registry UI 또는 Python SDK를 사용하여 컬렉션에 대한 설명을 대화형으로 또는 프로그래밍 방식으로 추가합니다.

{{< tabpane text=true >}}
  {{% tab header="W&B Registry UI" %}}
1. [https://wandb.ai/registry/](https://wandb.ai/registry/)의 W&B Registry로 이동합니다.
2. 컬렉션을 클릭합니다.
3. 컬렉션 이름 옆에 있는 **세부 정보 보기**를 선택합니다.
4. **설명** 필드 내에서 컬렉션에 대한 정보를 제공합니다. [Markdown 마크업 언어](https://www.markdownguide.org/)를 사용하여 텍스트 형식을 지정합니다.

  {{% /tab %}}
  {{% tab header="Python SDK" %}}

[`wandb.Api().artifact_collection()`]({{< relref path="/ref/python/public-api/api.md#artifact_collection" lang="ko" >}}) 메서드를 사용하여 컬렉션의 설명에 엑세스합니다. 반환된 오브젝트의 `description` 속성을 사용하여 컬렉션에 대한 설명을 추가하거나 업데이트합니다.

`type_name` 파라미터에 컬렉션의 유형을 지정하고 `name` 파라미터에 컬렉션의 전체 이름을 지정합니다. 컬렉션 이름은 접두사 "wandb-registry", 레지스트리 이름 및 컬렉션 이름으로 구성되며 슬래시로 구분됩니다.

```text
wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}
```

다음 코드 조각을 Python 스크립트 또는 노트북에 복사하여 붙여 넣습니다. 꺾쇠 괄호(`<>`)로 묶인 값을 자신의 값으로 바꿉니다.

```python
import wandb

api = wandb.Api()

collection = api.artifact_collection(
  type_name = "<collection_type>", 
  name = "<collection_name>"
  )


collection.description = "This is a description."
collection.save()  
```  
  {{% /tab %}}
{{< /tabpane >}}

예를 들어 다음 이미지는 모델 아키텍처, 용도, 성능 정보 등을 문서화하는 컬렉션을 보여줍니다.

{{< img src="/images/registry/registry_card.png" alt="모델 아키텍처, 용도, 성능 정보 등에 대한 정보가 포함된 컬렉션 카드입니다." >}}

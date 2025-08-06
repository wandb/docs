---
title: 컬렉션에 주석 달기
menu:
  default:
    identifier: ko-guides-core-registry-registry_cards
    parent: registry
weight: 8
---

컬렉션에 사용자가 컬렉션과 그 안에 포함된 Artifacts 의 목적을 이해할 수 있도록 읽기 쉬운 설명을 추가하세요.

컬렉션의 종류에 따라 트레이닝 데이터, 모델 아키텍처, 태스크, 라이선스, 참고문헌, 배포 등에 대한 정보를 포함할 수 있습니다. 아래는 컬렉션에서 문서화할 만한 주제를 안내합니다.

W&B 에서는 최소한 다음과 같은 세부 정보를 포함할 것을 권장합니다:
* **요약**: 컬렉션의 목적. 기계학습 실험에 사용한 기계학습 프레임워크 정보 등.
* **라이선스**: 기계학습 모델 사용에 적용되는 법적 조건과 권한. 사용자가 모델을 어떻게 활용할 수 있는지 이해하는 데 도움을 줍니다. 대표적인 라이선스로는 Apache 2.0, MIT, GPL 등이 있습니다.
* **참고자료**: 관련 논문, 데이터셋, 외부 리소스에 대한 인용 및 참고 정보 등.

컬렉션에 트레이닝 데이터가 포함된 경우, 다음과 같은 추가 정보를 함께 포함해보세요:
* **트레이닝 데이터**: 사용된 트레이닝 데이터에 대한 설명
* **처리 과정**: 트레이닝 데이터셋에 적용된 처리 방법
* **데이터 저장 위치**: 데이터가 어디에 저장되어 있으며, 어떻게 엑세스할 수 있는지 안내

컬렉션에 기계학습 모델이 포함된 경우, 다음과 같은 추가 정보를 함께 포함해보세요:
* **아키텍처**: 모델의 아키텍처, 레이어, 그리고 특별한 설계 결정 사항 등
* **태스크**: 컬렉션 모델이 수행하도록 설계된 특정 작업이나 문제. 모델의 용도를 분류하는 정보입니다.
* **모델 역직렬화 방법**: 팀원이 모델을 메모리에 적재할 수 있는 방법에 대한 안내
* **배포**: 모델의 배포 위치 및 방법, 워크플로우 오케스트레이션 플랫폼 등 엔터프라이즈 시스템과 통합하는 방법 등

## 컬렉션에 설명 추가하기

W&B Registry UI 또는 Python SDK를 사용하여 컬렉션에 설명을 직접 또는 프로그래밍 방식으로 추가할 수 있습니다.

{{< tabpane text=true >}}
  {{% tab header="W&B Registry UI" %}}
1. [W&B Registry App](https://wandb.ai/registry/)으로 이동합니다.
2. 컬렉션을 클릭합니다.
3. 컬렉션 이름 옆의 **View details**를 선택합니다.
4. **Description** 필드에 컬렉션에 대한 정보를 입력하세요. 텍스트는 [Markdown 마크업 언어](https://www.markdownguide.org/)로 포맷할 수 있습니다.

  {{% /tab %}}
  {{% tab header="Python SDK" %}}

[`wandb.Api().artifact_collection()`]({{< relref path="/ref/python/public-api/api.md#artifact_collection" lang="ko" >}}) 메소드를 사용해 컬렉션의 설명에 엑세스할 수 있습니다. 반환된 오브젝트의 `description` 속성을 활용하여 컬렉션에 설명을 추가하거나 업데이트할 수 있습니다.

`type_name` 파라미터에는 컬렉션의 타입을, `name` 파라미터에는 컬렉션의 전체 이름을 입력합니다. 전체 이름은 "wandb-registry" 접두사와 Registry 이름, Collection 이름이 슬래시(`/`)로 구분되어 구성됩니다:

```text
wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}
```

아래 코드조각을 Python 스크립트 또는 노트북에 복사하여 붙여넣고, 꺾쇠 괄호(`< >`)로 둘러싸인 값을 본인 환경에 맞게 수정하세요.

```python
import wandb

api = wandb.Api()

collection = api.artifact_collection(
  type_name = "<collection_type>", 
  name = "<collection_name>"
  )

# 컬렉션 설명 작성 및 저장
collection.description = "This is a description."
collection.save()  
```  
  {{% /tab %}}
{{< /tabpane >}}

예를 들어, 아래 이미지는 모델의 아키텍처, 예상 사용처, 성능 정보 등 다양한 항목이 문서화된 컬렉션을 보여줍니다.

{{< img src="/images/registry/registry_card.png" alt="Collection card" >}}
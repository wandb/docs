---
title: Create a collection
menu:
  default:
    identifier: ko-guides-models-registry-create_collection
    parent: registry
weight: 4
---

컬렉션은 레지스트리 내에서 연결된 아티팩트 버전의 *모음*입니다. 각 컬렉션은 고유한 작업 또는 유스 케이스를 나타냅니다.

예를 들어, 핵심 Dataset 레지스트리 내에 여러 컬렉션이 있을 수 있습니다. 각 컬렉션에는 MNIST, CIFAR-10 또는 ImageNet과 같은 서로 다른 데이터셋이 포함되어 있습니다.

또 다른 예로, "chatbot"이라는 레지스트리가 있고, 모델 Artifacts에 대한 컬렉션, 데이터셋 Artifacts에 대한 또 다른 컬렉션, 파인튜닝된 모델 Artifacts에 대한 또 다른 컬렉션을 포함할 수 있습니다.

레지스트리와 해당 컬렉션을 구성하는 방법은 사용자에게 달려 있습니다.

{{% alert %}}
W&B Model Registry에 익숙하신 분들은 등록된 모델에 대해 알고 계실 것입니다. Model Registry의 등록된 모델은 이제 W&B Registry에서 컬렉션이라고 합니다.
{{% /alert %}}

## 컬렉션 유형

각 컬렉션은 아티팩트의 *유형*을 하나만 허용합니다. 지정하는 유형은 귀하와 조직의 다른 구성원이 해당 컬렉션에 연결할 수 있는 아티팩트 종류를 제한합니다.

{{% alert %}}
아티팩트 유형을 Python과 같은 프로그래밍 언어의 데이터 유형과 유사하게 생각할 수 있습니다. 이 비유에서 컬렉션은 문자열, 정수 또는 부동 소수점을 저장할 수 있지만 이러한 데이터 유형의 혼합은 저장할 수 없습니다.
{{% /alert %}}

예를 들어 "dataset" 아티팩트 유형을 허용하는 컬렉션을 만든다고 가정합니다. 즉, "dataset" 유형의 향후 아티팩트 버전만 이 컬렉션에 연결할 수 있습니다. 마찬가지로, 모델 아티팩트 유형만 허용하는 컬렉션에는 "model" 유형의 아티팩트만 연결할 수 있습니다.

{{% alert %}}
아티팩트 오브젝트를 생성할 때 아티팩트의 유형을 지정합니다. `wandb.Artifact()`의 `type` 필드를 참고하세요.

```python
import wandb

# run 초기화
run = wandb.init(
  entity = "<team_entity>",
  project = "<project>"
  )

# 아티팩트 오브젝트 생성
artifact = wandb.Artifact(
    name="<artifact_name>", 
    type="<artifact_type>"
    )
```
{{% /alert %}}
 

컬렉션을 만들 때 미리 정의된 아티팩트 유형 목록에서 선택할 수 있습니다. 사용 가능한 아티팩트 유형은 컬렉션이 속한 레지스트리에 따라 다릅니다.

아티팩트를 컬렉션에 연결하거나 새 컬렉션을 만들기 전에 [컬렉션이 허용하는 아티팩트 유형을 조사]({{< relref path="#check-the-types-of-artifact-that-a-collection-accepts" lang="ko" >}})하세요.

### 컬렉션이 허용하는 아티팩트 유형 확인

컬렉션에 연결하기 전에 컬렉션이 허용하는 아티팩트 유형을 검사하세요. W&B Python SDK를 사용하여 프로그래밍 방식으로 또는 W&B App을 사용하여 대화식으로 컬렉션이 허용하는 아티팩트 유형을 검사할 수 있습니다.

{{% alert %}}
해당 아티팩트 유형을 허용하지 않는 컬렉션에 아티팩트를 연결하려고 하면 오류 메시지가 나타납니다.
{{% /alert %}}

{{< tabpane text=true >}}
  {{% tab header="W&B App" %}}
홈페이지의 레지스트리 카드 또는 레지스트리의 설정 페이지에서 허용된 아티팩트 유형을 찾을 수 있습니다.

두 방법 모두 먼저 W&B Registry App으로 이동합니다.

Registry App의 홈페이지 내에서 해당 레지스트리의 레지스트리 카드로 스크롤하여 허용된 아티팩트 유형을 볼 수 있습니다. 레지스트리 카드 내의 회색 가로 타원은 해당 레지스트리가 허용하는 아티팩트 유형을 나열합니다.

{{< img src="/images/registry/artifact_types_model_card.png" alt="" >}}

예를 들어, 앞의 이미지는 Registry App 홈페이지에 있는 여러 레지스트리 카드를 보여줍니다. **Model** 레지스트리 카드 내에서 **model** 및 **model-new**의 두 가지 아티팩트 유형을 볼 수 있습니다.

레지스트리의 설정 페이지 내에서 허용된 아티팩트 유형을 보려면:

1. 설정을 보려는 레지스트리 카드를 클릭합니다.
2. 오른쪽 상단 모서리에 있는 톱니바퀴 아이콘을 클릭합니다.
3. **허용된 아티팩트 유형** 필드로 스크롤합니다.
  {{% /tab %}}
  {{% tab header="Python SDK (Beta)" %}}
W&B Python SDK를 사용하여 레지스트리가 허용하는 아티팩트 유형을 프로그래밍 방식으로 봅니다.

```python
import wandb

registry_name = "<registry_name>"
artifact_types = wandb.Api().project(name=f"wandb-registry-{registry_name}").artifact_types()
print(artifact_type.name for artifact_type in artifact_types)
```

{{% alert %}}
이전 코드 조각으로 run을 초기화하지 마세요. 실험, 아티팩트 등을 추적하지 않고 W&B API를 쿼리만 하는 경우에는 run을 만들 필요가 없기 때문입니다.
{{% /alert %}}
  {{% /tab %}}
{{< /tabpane >}}

컬렉션이 허용하는 아티팩트 유형을 알게 되면 [컬렉션 만들기]({{< relref path="#create-a-collection" lang="ko" >}})를 할 수 있습니다.

## 컬렉션 만들기

레지스트리 내에서 대화식으로 또는 프로그래밍 방식으로 컬렉션을 만듭니다. 컬렉션을 만든 후에는 컬렉션이 허용하는 아티팩트 유형을 변경할 수 없습니다.

### 프로그래밍 방식으로 컬렉션 만들기

`wandb.init.link_artifact()` 메소드를 사용하여 아티팩트를 컬렉션에 연결합니다. `target_path` 필드에 컬렉션과 레지스트리를 모두 다음 형식의 경로로 지정합니다.

```python
f"wandb-registry-{registry_name}/{collection_name}"
```

여기서 `registry_name`은 레지스트리의 이름이고 `collection_name`은 컬렉션의 이름입니다. `wandb-registry-` 접두사를 레지스트리 이름에 추가해야 합니다.

{{% alert %}}
아티팩트를 존재하지 않는 컬렉션에 연결하려고 하면 W&B가 자동으로 컬렉션을 만듭니다. 존재하는 컬렉션을 지정하면 W&B는 아티팩트를 기존 컬렉션에 연결합니다.
{{% /alert %}}

이전 코드 조각은 프로그래밍 방식으로 컬렉션을 만드는 방법을 보여줍니다. `<>`로 묶인 다른 값을 자신의 값으로 바꾸세요.

```python
import wandb

# run 초기화
run = wandb.init(entity = "<team_entity>", project = "<project>")

# 아티팩트 오브젝트 생성
artifact = wandb.Artifact(
  name = "<artifact_name>",
  type = "<artifact_type>"
  )

registry_name = "<registry_name>"
collection_name = "<collection_name>"
target_path = f"wandb-registry-{registry_name}/{collection_name}"

# 컬렉션에 아티팩트 연결
run.link_artifact(artifact = artifact, target_path = target_path)

run.finish()
```

### 대화식으로 컬렉션 만들기

다음 단계는 W&B Registry App UI를 사용하여 레지스트리 내에서 컬렉션을 만드는 방법을 설명합니다.

1. W&B App UI에서 **Registry** App으로 이동합니다.
2. 레지스트리를 선택합니다.
3. 오른쪽 상단 모서리에 있는 **컬렉션 만들기** 버튼을 클릭합니다.
4. **이름** 필드에 컬렉션 이름을 입력합니다.
5. **유형** 드롭다운에서 유형을 선택합니다. 또는 레지스트리가 사용자 지정 아티팩트 유형을 활성화하는 경우 이 컬렉션이 허용하는 아티팩트 유형을 하나 이상 입력합니다.
6. 선택적으로 **설명** 필드에 컬렉션에 대한 설명을 입력합니다.
7. 선택적으로 **태그** 필드에 하나 이상의 태그를 추가합니다.
8. **버전 연결**을 클릭합니다.
9. **프로젝트** 드롭다운에서 아티팩트가 저장된 프로젝트를 선택합니다.
10. **아티팩트** 컬렉션 드롭다운에서 아티팩트를 선택합니다.
11. **버전** 드롭다운에서 컬렉션에 연결할 아티팩트 버전을 선택합니다.
12. **컬렉션 만들기** 버튼을 클릭합니다.

{{< img src="/images/registry/create_collection.gif" alt="" >}}

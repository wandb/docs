---
title: Migrate from legacy Model Registry
menu:
  default:
    identifier: ko-guides-models-registry-model_registry_eol
    parent: registry
weight: 8
---

W&B는 레거시 [W&B Model Registry]({{< relref path="/guides/models/registry/model_registry/" lang="ko" >}})의 에셋을 새로운 [W&B Registry]({{< relref path="./" lang="ko" >}})로 이전합니다. 이 마이그레이션은 W&B에서 완전히 관리하고 트리거하며, 사용자의 개입은 필요하지 않습니다. 이 프로세스는 기존 워크플로우를 최소한으로 중단하면서 최대한 원활하게 진행되도록 설계되었습니다.

이전은 새로운 W&B Registry에 Model Registry에서 현재 사용 가능한 모든 기능이 포함되면 시작됩니다. W&B는 현재 워크플로우, 코드 베이스 및 참조를 보존하려고 시도합니다.

이 가이드는 라이브 문서이며 더 많은 정보가 제공되면 정기적으로 업데이트됩니다. 질문이나 지원이 필요하면 support@wandb.com으로 문의하십시오.

## W&B Registry가 레거시 Model Registry와 다른 점

W&B Registry는 모델, 데이터셋 및 기타 Artifacts 관리를 위한 더욱 강력하고 유연한 환경을 제공하도록 설계된 다양한 새로운 기능과 개선 사항을 도입했습니다.

{{% alert %}}
레거시 Model Registry를 보려면 W&B App에서 Model Registry로 이동하십시오. 페이지 상단에 레거시 Model Registry App UI를 사용할 수 있는 배너가 나타납니다.

{{< img src="/images/registry/nav_to_old_model_reg.gif" >}}
{{% /alert %}}

### 조직 가시성
레거시 Model Registry에 연결된 Artifacts는 팀 수준의 가시성을 가집니다. 즉, 팀 멤버만이 레거시 W&B Model Registry에서 Artifacts를 볼 수 있습니다. W&B Registry는 조직 수준의 가시성을 가집니다. 즉, 올바른 권한을 가진 조직의 멤버는 registry에 연결된 Artifacts를 볼 수 있습니다.

### registry에 대한 가시성 제한
사용자 정의 registry를 보고 엑세스할 수 있는 사람을 제한합니다. 사용자 정의 registry를 만들 때 또는 사용자 정의 registry를 만든 후에 registry에 대한 가시성을 제한할 수 있습니다. 제한된 registry에서는 선택한 멤버만 콘텐츠에 엑세스할 수 있으므로 개인 정보 보호 및 제어가 유지됩니다. registry 가시성에 대한 자세한 내용은 [Registry 가시성 유형]({{< relref path="./configure_registry.md#registry-visibility-types" lang="ko" >}})을 참조하십시오.

### 사용자 정의 registries 만들기
레거시 Model Registry와 달리 W&B Registry는 Models 또는 데이터셋 registries에만 국한되지 않습니다. 특정 워크플로우 또는 프로젝트 요구 사항에 맞게 조정된 사용자 정의 registries를 만들어 임의의 오브젝트 유형을 담을 수 있습니다. 이러한 유연성을 통해 팀은 고유한 요구 사항에 따라 Artifacts를 구성하고 관리할 수 있습니다. 사용자 정의 registry를 만드는 방법에 대한 자세한 내용은 [사용자 정의 registry 만들기]({{< relref path="./create_registry.md" lang="ko" >}})를 참조하십시오.

{{< img src="/images/registry/mode_reg_eol.png" alt="" >}}

### 사용자 정의 엑세스 제어
각 registry는 멤버에게 관리자, 멤버 또는 뷰어와 같은 특정 역할을 할당할 수 있는 자세한 엑세스 제어를 지원합니다. 관리자는 멤버 추가 또는 제거, 역할 설정 및 가시성 구성을 포함한 registry 설정을 관리할 수 있습니다. 이를 통해 팀은 registries에서 Artifacts를 보고, 관리하고, 상호 작용할 수 있는 사용자를 제어할 수 있습니다.

{{< img src="/images/registry/registry_access_control.png" alt="" >}}

### 용어 업데이트
Registered models는 이제 *컬렉션*이라고 합니다.

### 변경 사항 요약

|               | 레거시 W&B Model Registry | W&B Registry |
| -----         | ----- | ----- |
| 아티팩트 가시성| 팀 멤버만 아티팩트를 보거나 액세스할 수 있습니다. | 조직의 멤버는 올바른 권한으로 레지스트리에 연결된 아티팩트를 보거나 액세스할 수 있습니다. |
| 사용자 정의 액세스 제어 | 사용할 수 없음 | 사용 가능 |
| 사용자 정의 레지스트리 | 사용할 수 없음 | 사용 가능 |
| 용어 업데이트 | 모델 버전 포인터(링크) 집합을 *등록된 모델*이라고 합니다. | 아티팩트 버전 포인터(링크) 집합을 *컬렉션*이라고 합니다. |
| `wandb.init.link_model` | Model Registry 특정 API | 현재는 레거시 모델 레지스트리하고만 호환됩니다. |

## 마이그레이션 준비

W&B는 등록된 Models (현재 컬렉션이라고 함)와 관련 Artifacts 버전을 레거시 Model Registry에서 W&B Registry로 마이그레이션합니다. 이 프로세스는 자동으로 수행되며 사용자의 조치가 필요하지 않습니다.

### 팀 가시성에서 조직 가시성으로

마이그레이션 후 모델 registry는 조직 수준의 가시성을 갖게 됩니다. [역할을 할당]({{< relref path="./configure_registry.md" lang="ko" >}})하여 registry에 엑세스할 수 있는 사용자를 제한할 수 있습니다. 이렇게 하면 특정 멤버만 특정 registries에 엑세스할 수 있습니다.

마이그레이션은 레거시 W&B Model Registry에서 현재 팀 수준으로 등록된 Models (곧 컬렉션이라고 함)의 기존 권한 경계를 보존합니다. 레거시 Model Registry에서 현재 정의된 권한은 새 Registry에서 보존됩니다. 즉, 현재 특정 팀 멤버로 제한된 컬렉션은 마이그레이션 중과 후에 보호됩니다.

### Artifacts 경로 연속성

현재 필요한 조치는 없습니다.

## 마이그레이션 중

W&B는 마이그레이션 프로세스를 시작합니다. 마이그레이션은 W&B 서비스에 대한 중단을 최소화하는 시간 내에 발생합니다. 레거시 Model Registry는 마이그레이션이 시작되면 읽기 전용 상태로 전환되며 참조용으로 엑세스할 수 있습니다.

## 마이그레이션 후

마이그레이션 후 컬렉션, Artifacts 버전 및 관련 속성을 새로운 W&B Registry 내에서 완전히 엑세스할 수 있습니다. 현재 워크플로우가 손상되지 않도록 유지하는 데 중점을 두고 있으며 변경 사항을 탐색하는 데 도움이 되는 지속적인 지원을 제공합니다.

### 새로운 registry 사용

사용자는 W&B Registry에서 사용할 수 있는 새로운 기능과 기능을 탐색하는 것이 좋습니다. Registry는 현재 사용되는 기능을 지원할 뿐만 아니라 사용자 정의 registries, 향상된 가시성 및 유연한 엑세스 제어와 같은 개선 사항도 도입합니다.

W&B Registry를 조기에 사용해 보거나 레거시 W&B Model Registry가 아닌 Registry로 시작하려는 신규 사용자에게 지원을 제공합니다. 이 기능을 활성화하려면 support@wandb.com 또는 Sales MLE에 문의하십시오. 초기 마이그레이션은 베타 버전으로 진행됩니다. W&B Registry의 베타 버전에는 레거시 Model Registry의 모든 기능이나 특징이 없을 수 있습니다.

자세한 내용과 W&B Registry의 전체 기능 범위에 대해 알아보려면 [W&B Registry 가이드]({{< relref path="./" lang="ko" >}})를 방문하십시오.

## FAQ

#### W&B가 Model Registry에서 W&B Registry로 에셋을 마이그레이션하는 이유는 무엇입니까?

W&B는 새로운 Registry를 통해 더욱 발전된 기능과 기능을 제공하기 위해 플랫폼을 발전시키고 있습니다. 이 마이그레이션은 Models, 데이터셋 및 기타 Artifacts 관리를 위한 더욱 통합되고 강력한 툴셋을 제공하기 위한 단계입니다.

#### 마이그레이션 전에 해야 할 일은 무엇입니까?

마이그레이션 전에 사용자가 해야 할 일은 없습니다. W&B는 전환을 처리하여 워크플로우와 참조가 보존되도록 합니다.

#### 모델 Artifacts에 대한 엑세스가 손실됩니까?

아니요, 마이그레이션 후에도 모델 Artifacts에 대한 엑세스는 유지됩니다. 레거시 Model Registry는 읽기 전용 상태로 유지되며 모든 관련 데이터가 새로운 Registry로 마이그레이션됩니다.

#### Artifacts 관련 메타데이터가 보존됩니까?

예, Artifacts 생성, 계보 및 기타 속성과 관련된 중요한 메타데이터는 마이그레이션 중에 보존됩니다. 사용자는 마이그레이션 후에도 모든 관련 메타데이터에 계속 엑세스할 수 있으므로 Artifacts의 무결성과 추적 가능성이 유지됩니다.

#### 도움이 필요하면 누구에게 문의해야 합니까?

질문이나 우려 사항이 있으면 지원을 받을 수 있습니다. 지원이 필요하면 support@wandb.com으로 문의하십시오.

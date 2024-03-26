---
title: Artifact automations
description: Use an project scoped artifact automation in your project to trigger
  actions when aliases or versions in an artifact collection are created or changed.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 아티팩트 변경으로 CI/CD 이벤트 트리거하기

아티팩트가 변경될 때 자동화를 생성하십시오. 아티팩트 버전 관리에 대한 하류 작업을 자동화하고자 할 때 아티팩트 자동화를 사용합니다. 자동화를 생성하려면, 발생하길 원하는 [액션](#action-types)을 기반으로 한 [이벤트 유형](#event-types)을 정의하세요.

아티팩트 변경으로 트리거되는 자동화에 대한 몇 가지 일반적인 유스 케이스는 다음과 같습니다:

* 평가/보류 데이터셋의 새 버전이 업로드되면, Model Registry의 최고 트레이닝 모델을 사용하여 추론을 수행하고 성능 정보가 포함된 리포트를 생성하는 [런치 작업 트리거](#create-a-launch-automation)합니다.
* 트레이닝 데이터셋의 새 버전이 "프로덕션"으로 표시되면, 현재 최고 성능 모델의 설정으로 [재트레이닝 런치](#create-a-launch-automation) 작업을 트리거합니다.

:::info
아티팩트 자동화는 프로젝트에 한정됩니다. 이는 프로젝트 내의 이벤트만 아티팩트 자동화를 트리거한다는 것을 의미합니다.

이는 W&B Model Registry에서 생성된 자동화와 대조됩니다. Model Registry에서 생성된 자동화는 Model Registry의 범위에 있으며, [Model Registry](../model_registry/intro.md)에 연결된 모델 버전에서 수행된 이벤트에 따라 트리거됩니다. 모델 버전에 대한 자동화를 생성하는 방법에 대한 정보는 [Model Registry 챕터](../model_registry/intro.md)의 [모델 CI/CD를 위한 자동화](../model_registry/automation.md) 페이지를 참조하세요.
:::

## 이벤트 유형
*Event*는 W&B 에코시스템에서 발생하는 변경입니다. 프로젝트의 아티팩트 컬렉션에 대해 두 가지 다른 이벤트 유형을 정의할 수 있습니다: **A new version of an artifact is created in a collection** 와 **An artifact alias is added됨**.

:::tip
**A new version of an artifact is created in a collection** 이벤트 유형을 사용하여 아티팩트의 각 버전에 반복적인 작업을 적용합니다. 예를 들어, 새 데이터셋 아티팩트 버전이 생성될 때마다 자동으로 트레이닝 작업을 시작하는 자동화를 생성할 수 있습니다.

**An artifact alias is added** 이벤트 유형을 사용하여 특정 에일리어스가 아티팩트 버전에 적용될 때 활성화되는 자동화를 생성합니다. 예를 들어, 누군가가 "test-set-quality-check" 에일리어스를 아티팩트에 추가하면 해당 데이터셋에 대한 하류 처리를 트리거하는 액션을 트리거하는 자동화를 생성할 수 있습니다.
:::

## 액션 유형
액션은 일부 트리거로 인해 발생하는 반응적 변형(내부 또는 외부)입니다. 프로젝트의 아티팩트 컬렉션에 대한 이벤트에 응답하여 생성할 수 있는 두 가지 유형의 액션이 있습니다: 웹훅과 [W&B 런치 작업](../launch/intro.md).

* 웹훅: HTTP 요청으로 W&B에서 외부 웹 서버와 통신합니다.
* W&B Launch job: [Jobs](../launch/create-launch-job.md)은 데스크톱이나 EKS의 Kubernetes, Amazon SageMaker 등과 같은 외부 컴퓨팅 리소스에서 새로운 [runs](../runs/intro.md)을 빠르게 런치할 수 있는 재사용 가능하고 구성 가능한 런 템플릿입니다.

다음 섹션에서는 웹훅과 W&B Launch로 자동화를 생성하는 방법을 설명합니다.

## 웹훅 자동화 생성
W&B App UI를 사용하여 액션을 기반으로 웹훅을 자동화하십시오. 이를 위해 먼저 웹훅을 설정한 다음 웹훅 자동화를 구성합니다.

### 인증 또는 인가를 위한 비밀 추가
비밀은 자격증명, API 키, 비밀번호, 토큰 등과 같은 개인 문자열을 가리는 팀 수준의 변수입니다. W&B는 평문 내용을 보호하고자 하는 모든 문자열을 저장하기 위해 비밀을 사용하도록 권장합니다.

웹훅에서 비밀을 사용하려면 먼저 해당 비밀을 팀의 비밀 관리자에 추가해야 합니다.

:::info
* W&B Admins만 비밀을 생성, 편집 또는 삭제할 수 있습니다.
* 외부 서버가 비밀을 사용하지 않는 경우 이 섹션을 건너뛰십시오.
* 비밀은 Azure, GCP 또는 AWS 배포에서 [W&B 서버](../hosting/intro.md)를 사용하는 경우에도 사용할 수 있습니다. 다른 배포 유형을 사용하는 경우 W&B에서 비밀을 사용하는 방법에 대해 논의하려면 W&B 계정 팀에 연락하십시오.
:::

웹훅 자동화를 사용할 때 W&B가 생성하는 것이 좋은 비밀 유형은 다음과 같습니다:

* **Access tokens**: 보안 웹훅 요청을 돕기 위해 발신자를 인증합니다.
* **Secret**: 페이로드에서 전송된 데이터의 진위성과 무결성을 보장합니다.

웹훅을 생성하려면 아래 지침을 따르십시오:

1. W&B 앱 UI로 이동합니다.
2. **Team Settings**을 클릭합니다.
3. 페이지를 아래로 스크롤하여 **Team secrets** 섹션을 찾습니다.
4. **New secret** 버튼을 클릭합니다.
5. 모달이 나타납니다. **Secret name** 필드에 비밀의 이름을 제공합니다.
6. **Secret** 필드에 비밀을 추가합니다.
7. (선택 사항) 웹훅이 추가적인 비밀 키나 토큰을 인증하는 데 필요한 경우 5단계와 6단계를 반복하여 다른 비밀(예: 액세스 토큰)을 생성합니다.

웹훅을 구성할 때 웹훅 자동화에 사용하려는 비밀을 지정합니다. 자세한 내용은 [웹훅 구성](#configure-a-webhook) 섹션을 참조하십시오.

:::tip
비밀을 생성하면 `$`를 사용하여 W&B 워크플로우에서 해당 비밀에 액세스할 수 있습니다.
:::

### 웹훅 구성
웹훅을 사용하기 전에 먼저 W&B 앱 UI에서 해당 웹훅을 구성해야 합니다.

:::info
* W&B Admins만 W&B 팀을 위해 웹훅을 구성할 수 있습니다.
* 웹훅이 추가적인 비밀 키나 토큰을 인증하는 데 필요한 경우 이미 [하나 이상의 비밀을 생성했는지](#add-a-secret-for-authentication-or-authorization) 확인하십시오.
:::

1. W&B App UI로 이동합니다.
2. **Team Settings**을 클릭합니다.
4. 페이지를 아래로 스크롤하여 **Webhooks** 섹션을 찾습니다.
5. **New webhook** 버튼을 클릭합니다.
6. **Name** 필드에 웹훅의 이름을 제공합니다.
7. **URL** 필드에 웹훅의 엔드포인트 URL을 제공합니다.
8. (선택 사항) **Secret** 드롭다운 메뉴에서 웹훅 페이로드를 인증하는 데 사용하려는 비밀을 선택합니다.
9. (선택 사항) **Access token** 드롭다운 메뉴에서 발신자를 인증하는 데 사용하려는 액세스 토큰을 선택합니다.
9. (선택 사항) 웹훅을 인증하는 데 필요한 추가적인 비밀 키나 토큰을 선택합니다(예: 액세스 토큰).

:::note
POST 요청에서 비밀과 액세스 토큰이 지정된 위치는 [웹훅 문제 해결](#troubleshoot-your-webhook) 섹션을 참조하십시오.
:::

### 웹훅 추가
웹훅을 구성하고 (선택 사항으로) 비밀을 가지고 있다면, 프로젝트 워크스페이스로 이동합니다. 왼쪽 사이드바에서 **Automations** 탭을 클릭합니다.

1. **Event type** 드롭다운에서 [이벤트 유형](#event-types)을 선택합니다.
![](/images/artifacts/artifact_webhook_select_event.png)
2. **A new version of an artifact is created in a collection** 이벤트를 선택한 경우, 자동화가 응답해야 하는 아티팩트 컬렉션의 이름을 **Artifact collection** 드롭다운에서 제공합니다.
![](/images/artifacts/webhook_new_version_artifact.png)
3. **Action type** 드롭다운에서 **Webhooks**을 선택합니다.
4. **Next step** 버튼을 클릭합니다.
5. **Webhook** 드롭다운에서 웹훅을 선택합니다.
![](/images/artifacts/artifacts_webhooks_select_from_dropdown.png)
6. (선택 사항) JSON 표현식 편집기에서 페이로드를 제공합니다. 일반적인 유스 케이스 예제는 [예제 페이로드](#example-payloads) 섹션을 참조하십시오.
7. **Next step**를 클릭합니다.
8. **Automation name** 필드에 웹훅 자동화의 이름을 제공합니다.
![](/images/artifacts/artifacts_webhook_name_automation.png)
9. (선택 사항) 웹훅에 대한 설명을 제공합니다.
10. **Create automation** 버튼을 클릭합니다.

### 예제 페이로드

다음 탭은 일반적인 유스 케이스를 기반으로 하는 예제 페이로드를 보여줍니다. 예제 내에서는 페이로드 매개변수의 조건 객체를 참조하기 위해 다음 키를 사용합니다:
* `${event_type}` 액션을 트리거한 이벤트 유형을 나타냅니다.
* `${event_author}` 액션을 트리거한 사용자를 나타냅니다.
* `${artifact_version}` 액션을 트리거한 특정 아티팩트 버전을 나타냅니다. 아티팩트 인스턴스로 전달됩니다.
* `${artifact_version_string}` 액션을 트리거한 특정 아티팩트 버전을 나타냅니다. 문자열로 전달됩니다.
* `${artifact_collection_name}` 아티팩트 버전이 연결된 아티팩트 컬렉션의 이름을 나타냅니다.
* `${project_name}` 액션을 트리거한 변형을 소유한 프로젝트의 이름을 나타냅니다.
* `${entity_name}` 액션을 트리거한 변형을 소유한 엔티티의 이름을 나타냅니다.


<Tabs
  defaultValue="github"
  values={[
    {label: 'GitHub 저장소 디스패치', value: 'github'},
    {label: 'Microsoft Teams 알림', value: 'microsoft'},
    {label: 'Slack 알림', value: 'slack'},
  ]}>
  <TabItem value="github">

:::info
GHA 워크플로우를 트리거하기 위해 필요한 권한이 액세스 토큰에 설정되어 있는지 확인하십시오. 자세한 내용은 [이 GitHub 문서를 참조하세요](https://docs.github.com/en/rest/repos/repos?#create-a-repository-dispatch-event).
:::

  W&B에서 GitHub 액션을 트리거하기 위해 저장소 디스패치를 보냅니다. 예를 들어, `on` 키에 대해 저장소 디스패치를 트리거로 수락하는 워크플로우가 있다고 가정합니다:

  ```yaml
  on:
    repository_dispatch:
      types: BUILD_AND_DEPLOY
  ```

  저장소에 대한 페이로드는 다음과 같을 수 있습니다:

  ```json
  {
    "event_type": "BUILD_AND_DEPLOY",
    "client_payload": 
    {
      "event_author": "${event_author}",
      "artifact_version": "${artifact_version}",
      "artifact_version_string": "${artifact_version_string}",
      "artifact_collection_name": "${artifact_collection_name}",
      "project_name": "${project_name}",
      "entity_name": "${entity_name}"
      }
  }

  ```

:::note
웹훅 페이로드의 `event_type` 키는 GitHub 워크플로우 YAML 파일의 `types` 필드와 일치해야 합니다.
:::

  자동화가 구성된 이벤트 또는 모델 버전에 따라 렌더링된 템플릿 문자열의 내용과 위치는 다음과 같습니다:

  ```json
  ${event_type} --> "LINK_ARTIFACT" 또는 "ADD_ARTIFACT_ALIAS"
  ${event_author} --> "<wandb-user>"
  ${artifact_version} --> "wandb-artifact://_id/QXJ0aWZhY3Q6NTE3ODg5ODg3""
  ${artifact_version_string} --> "<entity>/<project_name>/<artifact_name>:<alias>"
  ${artifact_collection_name} --> "<artifact_collection_name>"
  ${project_name} --> "<project_name>"
  ${entity_name} --> "<entity>"
  ```

  템플릿 문자열을 사용하여 W&B
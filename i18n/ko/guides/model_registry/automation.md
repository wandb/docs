---
title: Model registry automations
description: Use an Automation for model CI (automated model evaluation pipelines)
  and model deployment.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 모델 레지스트리 변경으로 CI/CD 이벤트 트리거하기

자동 모델 테스팅 및 배포와 같은 워크플로우 단계를 트리거하는 자동화를 생성합니다. 자동화를 생성하려면, 발생하길 원하는 [액션](#action-types)을 [이벤트 타입](#event-types) 기반으로 정의합니다.

예를 들어, 등록된 모델의 새 버전을 추가할 때 자동으로 모델을 GitHub에 배포하는 트리거를 생성할 수 있습니다.

## 이벤트 타입
*이벤트*는 W&B 에코시스템에서 발생하는 변경사항입니다. 모델 레지스트리는 두 가지 이벤트 타입을 지원합니다: **등록된 모델에 새 아티팩트 연결** 및 **등록된 모델 버전에 새 에일리어스 추가**.

:::tip
새 모델 후보를 테스트하기 위해 **등록된 모델에 새 아티팩트 연결** 이벤트 타입을 사용하세요. **등록된 모델 버전에 새 에일리어스 추가** 이벤트 타입을 사용하여 워크플로우의 특별한 단계를 나타내는 에일리어스를 지정하세요. 예를 들어 `deploy`와 같이 새 모델 버전에 해당 에일리어스가 적용될 때마다 사용하십시오.
:::

## 액션 타입
액션은 어떤 트리거의 결과로 발생하는 반응적인 변화(내부 또는 외부)입니다. 모델 레지스트리에서 생성할 수 있는 두 가지 유형의 액션이 있습니다: [웹훅](#create-a-webhook-automation) 및 [W&B Launch 작업](../launch/intro.md).

* 웹훅: HTTP 요청을 통해 W&B에서 외부 웹 서버와 통신합니다.
* W&B Launch 작업: [작업](../launch/create-launch-job.md)은 새로운 [run](../runs/intro.md)을 빠르게 시작할 수 있는 재사용 가능하고 구성 가능한 run 템플릿입니다. 예를 들어, 데스크톱의 로컬 또는 EKS의 Kubernetes, Amazon SageMaker 등과 같은 외부 컴퓨팅 리소스에서 실행할 수 있습니다.


다음 섹션에서는 웹훅 및 W&B Launch를 사용하여 자동화를 생성하는 방법을 설명합니다.

## 웹훅 자동화 생성
W&B App UI를 사용하여 액션을 기반으로 웹훅 자동화를 자동화합니다. 이를 위해 먼저 웹훅을 설정한 다음, 웹훅 자동화를 구성해야 합니다.

### 인증 또는 승인을 위한 비밀 추가
비밀은 팀 수준 변수로, 자격증명, API 키, 비밀번호, 토큰 등과 같은 개인 문자열을 숨기는 데 사용할 수 있습니다. W&B는 평문 내용을 보호하고자 하는 모든 문자열을 저장하기 위해 비밀을 사용할 것을 권장합니다.

웹훅에서 비밀을 사용하려면, 먼저 해당 비밀을 팀의 비밀 관리자에 추가해야 합니다.

:::info
* W&B 관리자만 비밀을 생성, 편집 또는 삭제할 수 있습니다.
* HTTP POST 요청을 보내는 외부 서버가 비밀을 사용하지 않는 경우 이 섹션을 건너뛰세요.
* [W&B 서버](../hosting/intro.md)를 Azure, GCP 또는 AWS 배포에서 사용하는 경우에도 비밀을 사용할 수 있습니다. 다른 배포 유형을 사용하는 경우 W&B에서 비밀을 사용하는 방법에 대해 W&B 계정 팀과 연결하세요.
:::

웹훅 자동화를 사용할 때 W&B가 생성할 것을 제안하는 비밀의 두 가지 유형이 있습니다:

* **액세스 토큰**: 발신자를 승인하여 웹훅 요청을 보호하는 데 도움이 됩니다.
* **비밀**: 페이로드에서 전송된 데이터의 진정성과 무결성을 보장합니다.

웹훅을 생성하기 위한 아래 지침을 따르세요:

1. W&B App UI로 이동합니다.
2. **팀 설정**을 클릭합니다.
3. **팀 비밀** 섹션을 찾을 때까지 페이지를 아래로 스크롤합니다.
4. **새 비밀** 버튼을 클릭합니다.
5. 모달이 나타납니다. **비밀 이름** 필드에 비밀의 이름을 제공합니다.
6. **비밀** 필드에 비밀을 추가합니다.
7. (선택사항) 웹훅이 추가 비밀 키나 토큰을 인증하는 데 필요한 경우(예: 액세스 토큰과 같은) 다른 비밀을 생성하려면 5단계와 6단계를 반복하세요.

웹훅을 구성할 때 사용하려는 비밀을 지정하세요. 자세한 내용은 [웹훅 구성](#configure-a-webhook) 섹션을 참조하세요.

:::tip
비밀을 생성하면 `$`를 사용하여 W&B 워크플로우에서 해당 비밀에 액세스할 수 있습니다.
:::

:::caution
W&B 서버에서 비밀을 사용하는 경우 고려할 사항:

보안 요구사항을 충족하는 보안 조치를 구성하는 것은 당신의 책임입니다.

W&B는 AWS, GCP 또는 Azure에서 제공하는 클라우드 비밀 관리자의 W&B 인스턴스에 비밀을 저장할 것을 강력히 권장합니다. AWS, GCP 및 Azure에서 제공하는 비밀 관리자는 고급 보안 기능으로 구성되어 있습니다.

클러스터를 비밀 저장소의 백엔드로 사용하는 것은 권장하지 않습니다. AWS, GCP 또는 Azure에서 제공하는 클라우드 비밀 관리자(W&B 인스턴스)를 사용할 수 없고, 클러스터를 사용할 경우 발생할 수 있는 보안 취약점을 방지하는 방법을 이해하는 경우에만 Kubernetes 클러스터를 고려하세요.
:::

### 웹훅 구성
웹훅을 사용하기 전에 먼저 W&B App UI에서 해당 웹훅을 구성해야 합니다.

:::info
* W&B 관리자만 W&B 팀을 위해 웹훅을 구성할 수 있습니다.
* 웹훅이 추가 비밀 키나 토큰을 인증하는 데 필요한 경우 이미 [하나 이상의 비밀을 생성했는지](#add-a-secret-for-authentication-or-authorization) 확인하세요.
:::

1. W&B App UI로 이동합니다.
2. **팀 설정**을 클릭합니다.
4. **웹훅** 섹션을 찾을 때까지 페이지를 아래로 스크롤합니다.
5. **새 웹훅** 버튼을 클릭합니다.
6. **이름** 필드에 웹훅의 이름을 제공합니다.
7. **URL** 필드에 웹훅의 엔드포인트 URL을 제공합니다.
8. (선택사항) **비밀** 드롭다운 메뉴에서 웹훅 페이로드를 인증하는 데 사용하려는 비밀을 선택합니다.
9. (선택사항) **액세스 토큰** 드롭다운 메뉴에서 발신자를 승인하는 데 사용하려는 액세스 토큰을 선택합니다.
9. (선택사항) **액세스 토큰** 드롭다운 메뉴에서 웹훅을 인증하는 데 필요한 추가 비밀 키나 토큰을 선택합니다(예: 액세스 토큰과 같은).

:::note
POST 요청에서 비밀과 액세스 토큰이 지정된 위치를 보려면 [웹훅 문제 해결](#troubleshoot-your-webhook) 섹션을 참조하세요.
:::

### 웹훅 추가
웹훅을 구성하고 (선택적으로) 비밀이 있는 경우 [https://wandb.ai/registry/model](https://wandb.ai/registry/model)에서 모델 레지스트리 앱으로 이동합니다.

1. **이벤트 타입** 드롭다운에서 [이벤트 타입](#event-types)을 선택합니다.
![](/images/models/webhook_select_event.png)
2. (선택사항) **등록된 모델에 새 버전이 추가됨** 이벤트를 선택한 경우 **등록된 모델** 드롭다운에서 등록된 모델의 이름을 제공합니다.
![](/images/models/webhook_new_version_reg_model.png)
3. **액션 유형** 드롭다운에서 **웹훅**을 선택합니다.
4. **다음 단계** 버튼을 클릭합니다.
5. **웹훅** 드롭다운에서 웹훅을 선택합니다.
![](/images/models/webhooks_select_from_dropdown.png)
6. (선택사항) JSON 표현식 편집기에 페이로드를 제공합니다. 일반적인 유스 케이스 예제는 [예시 페이로드](#example-payloads) 섹션을 참조하세요.
7. **다음 단계**를 클릭합니다.
8. **자동화 이름** 필드에 웹훅 자동화의 이름을 제공합니다.
![](/images/models/webhook_name_automation.png)
9. (선택사항) 웹훅에 대한 설명을 제공합니다.
10. **자동화 생성** 버튼을 클릭합니다.

### 예시 페이로드

다음 탭은 일반적인 유스 케이스를 기반으로 한 예시 페이로드를 보여줍니다. 예제 내에서는 페이로드 파라미터의 조건 오브젝트를 참조하기 위해 다음 키를 언급합니다:
* `${event_type}` 액션을 트리거한 이벤트 유형을 나타냅니다.
* `${event_author}` 액션을 트리거한 사용자를 나타냅니다.
* `${artifact_version}` 액션을 트리거한 특정 아티팩트 버전을 나타냅니다. 아티팩트 인스턴스로 전달됩니다.
* `${artifact_version_string}` 액션을 트리거한 특정 아티팩트 버전을 나타냅니다. 문자열로 전달됩니다.
* `${artifact_collection_name}` 아티팩트 버전이 연결된 아티팩트 컬렉션의 이름을 나타냅니다.
* `${project_name}` 액션을 트리거한 변이를 소유한 프로젝트의 이름을 나타냅니다.
* `${entity_name}` 액션을 트리거한 변이를 소유한 엔티티의 이름을 나타냅니다.


<Tabs
  defaultValue="github"
  values={[
    {label: 'GitHub 저장소 디스패치', value: 'github'},
    {label: 'Microsoft Teams 알림', value: 'microsoft'},
    {label: 'Slack 알림', value: 'slack'},
  ]}>
  <TabItem value="github">

:::info
GHA 워크플로우를 트리거할 권한이 있는 액세스 토큰을 확인하세요. 자세한 내용은 [GitHub 문서](https://docs.github.com/en/rest/repos/repos?#create-a-repository-dispatch-event)를 참조하세요.
:::
  
  W&B에서 GitHub 액션을 트리거하기 위해 저장소 디스패치를 보냅니다. 예를 들어, `on` 키에 대한 트리거로 저장소 디스패치를 수락하는 워크플로우가 있다고 가정해 보세요:

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

  자동화가 구성된 이벤트 또는 모델 버전에 따라 렌더링된 템플릿 문자열의 내용과 위치는 다를 수 있습니다. `${event_type}`은 "LINK_ARTIFACT" 또는 "ADD_ARTIFACT_ALIAS"로 렌더링됩니다. 아래 예제 매핑을 참조하세요:

  ```json
  ${event_type} --> "LINK_ARTIFACT" 또는 "ADD_ARTIFACT_ALIAS"
  ${event_author} --> "<wandb-user>"
  ${artifact_version} --> "wandb-artifact://_id/QXJ0aWZhY3Q6NTE3ODg5ODg3""
  ${artifact_version_string} --> "<entity>/model-registry/<registered_model_name>:<alias>"
  ${artifact_collection_name} --> "<registered_model_name>"
  ${project_name} --> "model-registry"
  ${entity_name} --> "<entity>"
  ```

  템플릿 문자열을 사용하여 W&B에서 GitHub Actions 및 기타 도구로 동적으로 컨텍스트를 전달하세요. 해당 도구가 Python 스크립트를 호출할 수 있다면, [W&B API](../artifacts/download-and-use-an-artifact.md)를 통해 등록된 모델 아티팩트를 소비할 수 있습니다.

  저장소 디스패치에 대한 자세한 내용은 [GitHub Marketplace의 공식 문서](https://github.com/marketplace/actions/repository-dispatch)를 참조하세요.  

  </TabItem>
  <TabItem value="microsoft">

  ‘수신 웹훅'을 구성하여 Teams 채널의 웹훅 URL을 가져옵니다. 다음은 예시 페이로드입니다:
  
  ```json 
  {
  "@type": "MessageCard",
  "@context": "http://schema.org/extensions",
  "summary": "새 알림",
  "sections": [
    {
      "activityTitle": "WANDB에서의 알림",
      "text": "Teams 웹훅을 통해 보낸 예시 메시지입니다.",
      "facts
---
title: Model registry automations
description: 모델 CI(자동화된 모델 평가 파이프라인)와 모델 배포에 Automation을 사용하십시오.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

자동화 설정을 통해 워크플로우 단계를 트리거하세요. 자동화 모델 테스트 및 배포와 같은 작업을 자동화할 수 있습니다. 자동화를 만들려면 [액션](#action-types)을 정의하여 [이벤트 유형](#event-types)에 따라 발생할 작업을 지정하세요.

예를 들어, 등록된 모델의 새로운 버전을 추가할 때 자동으로 모델을 GitHub에 배포하는 트리거를 만들 수 있습니다.

:::info
자동화에 대한 튜토리얼을 찾고 계십니까?
1. [이 튜토리얼](https://wandb.ai/wandb/wandb-model-cicd/reports/Model-CI-CD-with-W-B--Vmlldzo0OTcwNDQw)은 모델 평가 및 배포를 위한 Github 액션을 트리거하도록 자동화를 설정하는 방법을 보여줍니다.
2. [이 비디오 시리즈](https://youtube.com/playlist?list=PLD80i8An1OEGECFPgY-HPCNjXgGu-qGO6&feature=shared)는 webhook 기본 사항과 W&B에서 설정하는 방법을 보여줍니다.
3. [이 데모](https://www.youtube.com/watch?v=s5CMj_w3DaQ)는 Sagemaker Endpoint에 모델을 배포하기 위한 자동화를 설정하는 방법을 자세히 설명합니다.
:::

## 이벤트 유형
*이벤트*는 W&B 에코시스템에서 발생하는 변경 사항입니다. 모델 레지스트리는 두 가지 이벤트 유형을 지원합니다: **등록된 모델에 새로운 아티팩트를 연결** 및 **등록된 모델 버전에 새로운 에일리어스를 추가**입니다.

모델 버전을 연결하는 방법에 대한 자세한 내용은 [모델 버전 연결](./link-model-version.md)을 참조하세요. 그리고 아티팩트 에일리어스에 대한 정보는 [맞춤형 에일리어스 생성](../artifacts/create-a-custom-alias.md)을 참조하세요.

:::tip
새로운 모델 후보를 테스트하기 위해 **등록된 모델에 새로운 아티팩트를 연결하는** 이벤트 유형을 사용하세요. 워크플로우의 특별한 단계, 예를 들어 `deploy`를 나타내는 에일리어스를 지정하기 위해 등록된 모델 버전에 새로운 에일리어스를 추가하는 이벤트 유형을 사용하세요. 이 에일리어스가 적용된 새 모델 버전이 있을 때마다 사용할 수 있습니다.
:::

## 액션 유형
액션은 일부 트리거의 결과로 발생하는 응답성 변형(내부 또는 외부)입니다. 모델 레지스트리에서 생성할 수 있는 두 가지 액션 유형은 [webhooks](#create-a-webhook-automation)과 [W&B Launch Jobs](../launch/intro.md)입니다.

* Webhooks: HTTP 요청을 통해 W&B에서 외부 웹 서버와 통신합니다.
* W&B Launch job: [Jobs](../launch/create-launch-job.md)는 데스크탑 또는 Kubernetes on EKS, Amazon SageMaker 등 외부 컴퓨팅 자원에서 새로운 [runs](../runs/intro.md)를 신속하게 시작할 수 있게 해주는 재사용 가능하고 구성 가능한 러닝 템플릿입니다.

다음 섹션에서는 Webhooks 및 W&B Launch를 사용하여 자동화를 생성하는 방법을 설명합니다.

## 웹훅 자동화 생성
W&B 애플리케이션 UI로 액션에 기반하여 웹훅을 자동화하세요. 이를 위해서는 먼저 웹훅을 설정하고, 그런 다음 웹훅 자동화를 구성해야 합니다.

:::info
웹훅에 어드레스 레코드(A 레코드)가 있는 엔드포인트를 지정하세요. W&B는 `[0-255].[0-255].[0-255].[0.255]`나 `localhost`처럼 IP 어드레스로 직접 노출되는 엔드포인트에 대한 연결을 지원하지 않습니다. 이러한 제한은 서버 측 요청 위조(SSRF) 공격 및 기타 관련된 위협 벡터로부터 보호하는 데 도움이 됩니다.
:::

### 인증 또는 권한 부여를 위한 비밀 추가
비밀은 자격 증명, API 키, 비밀번호, 토큰 등과 같은 개인 문자열을 숨길 수 있게 하는 팀 수준 변수입니다. W&B에서는 비밀을 사용하여 보호하고자 하는 모든 문자열을 저장할 것을 권장합니다.

웹훅에서 비밀을 사용하려면 먼저 해당 비밀을 팀의 비밀 관리자에 추가해야 합니다.

:::info
* 오직 W&B 관리자만이 비밀을 생성, 편집 또는 삭제할 수 있습니다.
* External server가 HTTP POST 요청을 보낼 때 비밀을 사용하지 않는 경우 이 섹션을 건너뛰세요. 
* 비밀은 또한 Azure, GCP 또는 AWS 배포에서 [W&B 서버](../hosting/intro.md)를 사용할 경우에도 제공됩니다. 다른 배포 유형을 사용하는 경우 W&B 계정 팀에 연락하여 W&B에서 비밀을 사용할 수 있는 방법을 논의하세요.
:::

웹훅 자동화를 사용할 때 W&B에서는 다음 두 가지 유형의 비밀을 생성할 것을 제안합니다:

* **Access tokens**: 발신자를 인증하여 웹훅 요청의 보안을 강화하세요.
* **Secret**: 페이로드에서 전송된 데이터의 진위성과 무결성을 보장하세요.

웹훅을 생성하려면 아래 지침을 따르세요:

1. W&B 앱 UI로 이동하세요.
2. **팀 설정**을 클릭하세요.
3. **팀 비밀** 섹션이 나타날 때까지 페이지를 아래로 스크롤하세요.
4. **새로운 비밀** 버튼을 클릭하세요.
5. 모달이 나타납니다. **비밀 이름** 필드에 비밀의 이름을 입력하세요.
6. **비밀** 필드에 비밀을 추가하세요.
7. (선택 사항) 웹훅에 추가적인 비밀 키 또는 토큰이 필요한 경우 다른 비밀(예: 액세스 토큰)을 생성하기 위해 단계 5와 6을 반복하세요.

웹훅을 구성할 때 웹훅 자동화에 사용할 비밀을 지정하세요. 자세한 내용은 [웹훅 구성](#configure-a-webhook) 섹션을 참조하세요.

:::tip
비밀을 생성하면 해당 비밀을 W&B 워크플로우에서 `$`를 사용하여 엑세스할 수 있습니다.
:::

:::caution
W&B 서버에서 비밀을 사용하는 경우 고려 사항:

보안 요구를 충족하는 보안 조치를 구성할 책임이 있습니다.

W&B는 AWS, GCP 또는 Azure의 클라우드 비밀 관리자가 제공하는 W&B 인스턴스에 비밀을 저장할 것을 강력히 권장합니다. AWS, GCP, Azure에서 제공하는 비밀 관리자는 고급 보안 기능으로 구성되어 있습니다.

백엔드 비밀 저장소로 Kubernetes 클러스터를 사용하는 것은 권장되지 않습니다. AWS, GCP, Azure와 같은 클라우드 비밀 관리자의 W&B 인스턴스를 사용할 수 없는 경우에만 Kubernetes 클러스터를 고려하시고, 클러스터를 사용 시 발생할 수 있는 보안 취약성을 예방하는 방법을 잘 알고 있어야 합니다.
:::

### 웹훅 구성
웹훅을 사용하기 전에 W&B 애플리케이션 UI에서 해당 웹훅을 먼저 구성해야 합니다.

:::info
* 오직 W&B 관리자만 W&B 팀에 대한 웹훅을 구성할 수 있습니다.
* 웹훅이 추가적인 비밀 키나 토큰을 사용하여 인증을 필요로하는 경우 [하나 이상의 비밀을 생성](#add-a-secret-for-authentication-or-authorization)했는지 확인하세요.
:::

1. W&B 앱 UI로 이동하세요.
2. **팀 설정**을 클릭하세요.
3. **Webhooks** 섹션이 나타날 때까지 페이지를 아래로 스크롤하세요.
4. **새 웹훅** 버튼을 클릭하세요.
5. **이름** 필드에 웹훅의 이름을 입력하세요.
6. **URL** 필드에 웹훅의 엔드포인트 URL을 제공하세요.
7. (선택 사항) **비밀** 드롭다운 메뉴에서 웹훅 페이로드 인증에 사용하고자 하는 비밀을 선택하세요.
8. (선택 사항) **액세스 토큰** 드롭다운 메뉴에서 발신자 인증을 위한 액세스 토큰을 선택하세요.
9. (선택 사항) 웹훅 인증에 필요한 추가 비밀 키 또는 토큰(예: 액세스 토큰)을 **액세스 토큰** 드롭다운 메뉴에서 선택하세요.

:::note
POST 요청에서 비밀 및 액세스 토큰이 지정되는 위치를 보려면 [웹훅 문제 해결](#troubleshoot-your-webhook) 섹션을 참조하세요.
:::

### 웹훅 추가
웹훅이 구성되고 (선택적으로) 비밀이 추가되면, [https://wandb.ai/registry/model](https://wandb.ai/registry/model)의 모델 레지스트리 앱으로 이동하세요.

1. **이벤트 유형** 드롭다운에서 [이벤트 유형](#event-types)을 선택하세요.
![](/images/models/webhook_select_event.png)
2. (선택 사항) **등록된 모델에 새 버전이 추가됨** 이벤트를 선택한 경우, **등록된 모델** 드롭다운에서 등록된 모델의 이름을 제공하세요.
![](/images/models/webhook_new_version_reg_model.png)
3. **액션 유형** 드롭다운에서 **Webhooks**를 선택하세요.
4. **다음 단계** 버튼을 클릭하세요.
5. **Webhook** 드롭다운에서 웹훅을 선택하세요.
![](/images/models/webhooks_select_from_dropdown.png)
6. (선택 사항) JSON 표현식 편집기에 페이로드를 제공하세요. 일반적인 유스 케이스 예제는 [예제 페이로드](#example-payloads) 섹션을 참조하세요.
7. **다음 단계**를 클릭하세요.
8. **자동화 이름** 필드에 웹훅 자동화의 이름을 제공하세요.
![](/images/models/webhook_name_automation.png)
9. (선택 사항) 웹훅에 대한 설명을 제공하세요.
10. **자동화 생성** 버튼을 클릭하세요.

### 예제 페이로드

다음 탭은 일반적인 유스 케이스를 기반으로 한 예제 페이로드를 보여줍니다. 예제 내에서는 페이로드 파라미터의 조건 오브젝트를 참조하기 위해 다음 키를 참조합니다:
* `${event_type}` 트리거된 액션의 이벤트 유형을 참조합니다.
* `${event_author}` 트리거된 액션의 사용자를 참조합니다.
* `${artifact_version}` 트리거된 액션의 특정 아티팩트 버전. 아티팩트 인스턴스로 전달됩니다.
* `${artifact_version_string}` 트리거된 액션의 특정 아티팩트 버전. 문자열로 전달됩니다.
* `${artifact_collection_name}` 아티팩트 버전이 연결된 아티팩트 컬렉션의 이름을 참조합니다.
* `${project_name}` 트리거된 변형의 소유 프로젝트의 이름을 참조합니다.
* `${entity_name}` 트리거된 변형의 소유 엔티티의 이름을 참조합니다.

<Tabs
  defaultValue="github"
  values={[
    {label: 'GitHub repository dispatch', value: 'github'},
    {label: 'Microsoft Teams notification', value: 'microsoft'},
    {label: 'Slack notifications', value: 'slack'},
  ]}>
  <TabItem value="github">

:::info
GitHub Actions 워크플로우를 트리거하기 위해 엑세스 토큰에 필요한 권한 세트가 있는지 확인하세요. 자세한 내용은 [GitHub 문서](https://docs.github.com/en/rest/repos/repos?#create-a-repository-dispatch-event)를 참조하세요.
:::

W&B에서 GitHub 액션을 트리거하기 위해 리포지토리 디스패치를 전송하세요. 예를 들어, `on` 키에 대한 트리거로 리포지토리 디스패치를 수락하는 워크플로우가 있다고 가정해보세요:

  ```yaml
  on:
    repository_dispatch:
      types: BUILD_AND_DEPLOY
  ```

  리포지토리에 대한 페이로드는 다음과 같을 수 있습니다:

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
GitHub 워크플로우 YAML 파일의 `types` 필드와 웹훅 페이로드의 `event_type` 키가 일치해야 합니다.
:::

자동화가 구성된 이벤트 또는 모델 버전에 따라 렌더링된 템플릿 문자열의 내용 및 위치가 결정됩니다. `${event_type}`는 "LINK_ARTIFACT" 또는 "ADD_ARTIFACT_ALIAS"로 렌더링됩니다. 아래는 예제 매핑입니다:

  ```json
  ${event_type} --> "LINK_ARTIFACT" or "ADD_ARTIFACT_ALIAS"
  ${event_author} --> "<wandb-user>"
  ${artifact_version} --> "wandb-artifact://_id/QXJ0aWZhY3Q6NTE3ODg5ODg7""
  ${artifact_version_string} --> "<entity>/model-registry/<등록된_모델_이름>:<에일리어스>"
  ${artifact_collection_name} --> "<등록된_모델_이름>"
  ${project_name} --> "model-registry"
  ${entity_name} --> "<entity>"
  ```

템플릿 문자열을 사용하여 W&B에서 GitHub Actions 및 다른 툴로 동적으로 컨텍스트를 전달합니다. 만약 이러한 툴들이 파이썬 스크립트를 호출할 수 있다면, 등록된 모델 아티팩트를 [W&B API](../artifacts/download-and-use-an-artifact.md)를 통해 수신할 수 있습니다.

리포지토리 디스패치에 대한 더 자세한 내용은 [GitHub Marketplace 공식 문서](https://github.com/marketplace/actions/repository-dispatch)를 참조하세요.

[Webhook Automations for Model Evaluation](https://www.youtube.com/watch?v=7j-Mtbo-E74&ab_channel=Weights%26Biases) 및 [Webhook Automations for Model Deployment](https://www.youtube.com/watch?v=g5UiAFjM2nA&ab_channel=Weights%26Biases) YouTube 비디오는 각각 모델 평가 및 배포를 위한 자동화를 생성하는 방법을 단계별로 보여줍니다.

모델 CI를 위해 Github Actions 웹훅 자동화를 사용하는 방법을 배우려면 이 W&B [리포트](https://wandb.ai/wandb/wandb-model-cicd/reports/Model-CI-CD-with-W-B--Vmlldzo0OTcwNDQw)를 참조하세요. Modal Labs 웹훅으로 모델 CI를 만드는 방법을 배우려면 이 [GitHub 리포지토리](https://github.com/hamelsmu/wandb-modal-webhook)를 참조하세요.

  </TabItem>
  <TabItem value="microsoft">

  Teams 채널에 대한 웹훅 URL을 생성하려면 'Incoming Webhook'을 구성하세요. 다음은 예제 페이로드입니다:
  
  ```json 
  {
  "@type": "MessageCard",
  "@context": "http://schema.org/extensions",
  "summary": "New Notification",
  "sections": [
    {
      "activityTitle": "Notification from WANDB",
      "text": "This is an example message sent via Teams webhook.",
      "facts": [
        {
          "name": "Author",
          "value": "${event_author}"
        },
        {
          "name": "Event Type",
          "value": "${event_type}"
        }
      ],
      "markdown": true
    }
  ]
  }
  ```
  이 Teams 예제처럼, 실행 시점에 W&B 데이터를 페이로드에 주입하기 위해 템플릿 문자열을 사용할 수 있습니다.

  </TabItem>
  <TabItem value="slack">

  Slack 앱을 설정하고 [Slack API 문서](https://api.slack.com/messaging/webhooks)에 강조된 지침을 따라 들어오는 웹훅 인테그레이션을 추가하세요. W&B 웹훅의 액세스 토큰으로 `Bot User OAuth Token`에 비밀이 지정되었는지 확인하세요. 
  
  다음은 예제 페이로드입니다:

  ```json
    {
        "text": "New alert from WANDB!",
    "blocks": [
        {
                "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "Registry event: ${event_type}"
            }
        },
            {
                "type":"section",
                "text": {
                "type": "mrkdwn",
                "text": "New version: ${artifact_version_string}"
            }
            },
            {
            "type": "divider"
        },
            {
                "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "Author: ${event_author}"
            }
            }
        ]
    }
  ```

  </TabItem>
</Tabs>

### 웹훅 문제 해결

웹훅을 W&B 앱 UI에서 인터랙티브하게 또는 바시 스크립트를 사용하여 프로그래밍적으로 문제를 해결하세요. 새로운 웹훅을 생성하거나 기존 웹훅을 수정할 때 웹훅을 문제 해결할 수 있습니다.

<Tabs
  defaultValue="app"
  values={[
    {label: 'W&B App UI', value: 'app'},
    {label: 'Bash script', value: 'bash'},
  ]}>
  <TabItem value="app">

W&B 앱 UI에서 인터랙티브하게 웹훅 테스트를 수행하세요.

1. W&B 팀 설정 페이지로 이동하세요.
2. **Webhooks** 섹션으로 스크롤하세요.
3. 웹훅 이름 옆의 세 개의 점이 있는 아이콘(미트볼 아이콘)을 클릭하세요.
4. **테스트**를 선택하세요.
5. 나타나는 UI 패널에서 POST 요청을 해당 필드에 붙여넣으세요.
![](/images/models/webhook_ui.png)
6. **웹훅 테스트**를 클릭하세요.

W&B 앱 UI 내에서, W&B는 엔드포인트가 응답한 내용을 게시합니다.

![](/images/models/webhook_ui_testing.gif)

실제 예제를 보려면 [Testing Webhooks in Weights & Biases](https://www.youtube.com/watch?v=bl44fDpMGJw&ab_channel=Weights%26Biases) YouTube 동영상을 참조하세요.

  </TabItem>
  <TabItem value="bash">

다음 bash 스크립트는 트리거될 때 W&B가 웹훅 자동화로 보내는 POST 요청과 유사한 POST 요청을 생성합니다.

아래 코드를 셸 스크립트에 복사하여 웹훅 문제를 해결하세요. 다음 항목의 값을 직접 지정하세요:

* `ACCESS_TOKEN`
* `SECRET`
* `PAYLOAD`
* `API_ENDPOINT`

```sh title="webhook_test.sh"
#!/bin/bash

# 귀하의 엑세스 토큰 및 비밀
ACCESS_TOKEN="your_api_key" 
SECRET="your_api_secret"

# 보낼 데이터(예: JSON 형식)
PAYLOAD='{"key1": "value1", "key2": "value2"}'

# HMAC 서명 생성
# 보안을 위해, Wandb는 헤더에 X-Wandb-Signature를 포함하여 페이로드와 웹훅과 연관된 공유 비밀 키로부터 HMAC과 SHA-256 알고리즘을 사용하여 계산합니다.
SIGNATURE=$(echo -n "$PAYLOAD" | openssl dgst -sha256 -hmac "$SECRET" -binary | base64)

# cURL 요청 실행
curl -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "X-Wandb-Signature: $SIGNATURE" \
  -d "$PAYLOAD" API_ENDPOINT
```

  </TabItem>
</Tabs>

## 자동 시작 자동화 생성

W&B Job을 자동으로 시작하세요.

:::info
이 섹션에서는 이미 job, queue를 생성하고 활성 에이전트가 폴링 중인 상태임을 가정합니다. 더 많은 정보는 [W&B Launch 문서](../launch/intro.md)를 참조하세요.
:::

1. **이벤트 유형** 드롭다운에서 이벤트 유형을 선택하세요. 지원되는 이벤트에 대한 정보는 [이벤트 유형](#event-types) 섹션을 참조하세요.
2. (선택 사항) **등록된 모델에 새 버전이 추가됨** 이벤트를 선택한 경우, **등록된 모델** 드롭다운에서 등록된 모델의 이름을 제공하세요.
3. **액션 유형** 드롭다운에서 **Jobs**를 선택하세요.
4. **Job** 드롭다운에서 W&B Launch Job을 선택하세요.  
5. **Job 버전** 드롭다운에서 버전을 선택하세요.
6. (선택 사항) 새 job에 하이퍼파라미터 오버라이드를 제공하세요.
7. **대상 프로젝트** 드롭다운에서 프로젝트를 선택하세요.
8. job을 큐에 추가할 큐를 선택하세요.  
9. **다음 단계**를 클릭하세요.
10. **자동화 이름** 필드에 웹훅 자동화의 이름을 제공하세요.
11. (선택 사항) 웹훅에 대한 설명을 제공합니다.
12. **자동화 생성** 버튼을 클릭하세요.

모델 CI를 위한 자동화를 W&B Launch를 사용하여 생성하는 방법에 대한 끝에서 끝에 이르는 예를 보려면 이 [예제 리포트](https://wandb.ai/examples/wandb_automations/reports/Model-CI-with-W-B-Automations--Vmlldzo0NDY5OTIx)를 참조하세요.

## 자동화 보기

W&B 앱 UI에서 등록된 모델과 연관된 자동화를 확인하세요.

1. [https://wandb.ai/registry/model](https://wandb.ai/registry/model)에서 모델 레지스트리 앱으로 이동하세요.
2. 등록된 모델을 선택하세요. 
3. 페이지 하단의 **Automations** 섹션으로 스크롤하세요.

Automations 섹션에서 선택한 모델에 대해 생성된 자동화의 다음 속성을 확인할 수 있습니다:

- **트리거 유형**: 구성된 트리거 유형.
- **액션 유형**: 자동화를 트리거하는 액션 유형입니다. 사용 가능한 옵션은 Webhooks 및 Launch입니다.
- **액션 이름**: 자동화를 생성할 때 제공한 액션 이름입니다.
- **Queue**: 잡이 큐에 추가된 이름. 웹훅 액션 유형을 선택한 경우 이 필드는 비어 있습니다.

## 자동화 삭제
모델과 연결된 자동화를 삭제합니다. 작업이 완료되기 전에 자동화를 삭제해도 진행 중인 작업에는 영향을 미치지 않습니다.

1. [https://wandb.ai/registry/model](https://wandb.ai/registry/model)에서 모델 레지스트리 앱으로 이동하세요.
2. 등록된 모델을 클릭하세요. 
3. 페이지 하단의 **Automations** 섹션으로 스크롤하세요.
4. 자동화 이름 옆에 마우스를 가져가 세 개의 세로점(kebob) 메뉴를 클릭하세요.
5. **삭제**를 선택하세요.
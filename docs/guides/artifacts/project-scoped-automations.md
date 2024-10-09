---
title: Trigger CI/CD events when artifact changes
description: 프로젝트 내에서 에일리어스나 버전이 아티팩트 컬렉션에 생성되거나 변경될 때 작업을 트리거하기 위해 프로젝트 범위의 아티팩트 자동화를 사용하세요.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

아티팩트가 변경되면 자동으로 트리거되는 자동화를 생성하세요. 아티팩트의 버전 관리에 관한 후속 작업을 자동화하려면 아티팩트 자동화를 사용하세요. 자동화를 만들려면 원하는 [액션](#action-types)을 정의하고 [이벤트 유형](#event-types)에 기반하여 발생하도록 설정하세요.

아티팩트 변경에 의해 트리거되는 자동화의 일반적인 유스 케이스에는 다음이 포함됩니다:

* 평가/홀드아웃 데이터셋의 새로운 버전이 업로드되면, 모델 레지스트리에서 가장 우수한 트레이닝 모델을 사용하여 추론을 수행하고 성능 정보를 담은 리포트를 생성하는 [Launch job을 트리거](#create-a-launch-automation)합니다.
* 트레이닝 데이터셋의 새로운 버전이 "프로덕션"으로 레이블링되면, 현재 가장 성능이 좋은 모델의 설정을 사용하여 재트레이닝 Launch job을 트리거합니다.

:::info
아티팩트 자동화는 프로젝트에 한정됩니다. 즉, 프로젝트 내의 이벤트만이 아티팩트 자동화를 트리거할 수 있습니다.

이는 W&B 모델 레지스트리에서 생성된 자동화와 대조됩니다. 모델 레지스트리 내에서 생성된 자동화는 모델 레지스트리에 있는 모델 버전에 수행된 이벤트가 있을 때 트리거됩니다. 모델 버전에 대한 자동화를 생성하는 방법에 대한 정보는 [Model CI/CD의 자동화](../model_registry/model-registry-automations.md) 페이지 및 [모델 레지스트리 챕터](../model_registry/intro.md)를 참조하세요.
:::

## 이벤트 유형
*이벤트*는 W&B 에코시스템에서 발생하는 변경 사항입니다. 프로젝트의 아티팩트 컬렉션에 대한 두 가지 다른 이벤트 유형을 정의할 수 있습니다: **컬렉션에서 아티팩트의 새로운 버전이 생성됨**과 **아티팩트 에일리어스가 추가됨**입니다.

:::tip
각 아티팩트 버전에 반복 작업을 적용하려면 **컬렉션에서 아티팩트의 새로운 버전이 생성됨** 이벤트 유형을 사용하세요. 예를 들어, 새로운 데이터셋 아티팩트 버전이 생성될 때 자동으로 트레이닝 job을 시작하는 자동화를 생성할 수 있습니다.

특정 에일리어스가 아티팩트 버전에 적용될 때 자동화를 활성화하려면 **아티팩트 에일리어스가 추가됨** 이벤트 유형을 사용하세요. 예를 들어, 누군가가 "test-set-quality-check" 에일리어스를 아티팩트에 추가하여 해당 데이터셋에 대해 후속 처리를 트리거할 때 액션을 트리거하는 자동화를 생성할 수 있습니다.
:::

## 액션 유형
액션은 일부 트리거가 발생한 결과로 일어나는 반응적 변경(내부 또는 외부)입니다. 프로젝트의 아티팩트 컬렉션에서 발생한 이벤트에 대한 응답으로 생성할 수 있는 두 가지 유형의 액션이 있습니다: 웹훅과 [W&B Launch Jobs](../launch/intro.md).

* 웹훅: HTTP 요청을 통해 W&B에서 외부 웹 서버와 통신합니다.
* W&B Launch job: [Jobs](../launch/create-launch-job.md)는 로컬 데스크탑 또는 Kubernetes on EKS, Amazon SageMaker 등의 외부 컴퓨팅 자원에서 빠르게 새로운 [runs](../runs/intro.md)를 시작할 수 있도록 하는 재사용 가능하고 구성 가능한 run 템플릿입니다.

다음 섹션에서는 웹훅 및 W&B Launch를 사용하여 자동화를 생성하는 방법에 대해 설명합니다.

## 웹훅 자동화 생성
W&B 앱 UI를 사용하여 액션에 기반한 웹훅을 자동화하세요. 이를 위해 먼저 웹훅을 설정한 후 웹훅 자동화를 구성합니다.

:::info
어드레스 레코드(A 레코드)를 가진 엔드포인트를 웹훅에 지정하세요. W&B는 `[0-255].[0-255].[0-255].[0-255]` 또는 `localhost`와 같이 IP 어드레스로 노출된 엔드포인트에 연결하는 것을 지원하지 않습니다. 이 제한은 서버 측 요청 위조(SSRF) 공격 및 관련 위협 벡터로부터 보호하기 위함입니다.
:::

### 인증 또는 권한 부여를 위한 시크릿 추가
시크릿은 자격 증명, API 키, 비밀번호, 토큰 등의 비공개 문자열을 모호하게 할 수 있도록 하는 팀 수준의 변수입니다. W&B는 평문 내용 보호가 필요한 문자열을 저장할 때 시크릿을 사용할 것을 권장합니다.

웹훅에서 시크릿을 사용하려면 먼저 해당 시크릿을 팀의 시크릿 관리자에 추가해야 합니다.

:::info
* W&B 관리자가 아닌 경우에는 시크릿을 생성, 편집 또는 삭제할 수 없습니다.
* HTTP POST 요청을 보내는 외부 서버가 시크릿을 사용하지 않는다면 이 섹션을 건너뛰세요.
* [W&B 서버](../hosting/intro.md)를 Azure, GCP 또는 AWS 배포에서 사용하신다면 시크릿도 사용할 수 있습니다. 다른 배포 유형을 사용하신다면 W&B 계정 팀에 문의하여 W&B에서 시크릿을 사용하는 방법을 논의하세요.
:::

웹훅 자동화를 사용할 때 W&B가 권장하는 두 가지 유형의 시크릿이 있습니다:

* **엑세스 토큰**: 웹훅 요청의 보안을 강화하기 위해 발신자를 인증합니다.
* **시크릿**: 페이로드로 전송된 데이터의 진본성과 무결성을 보장합니다.

다음 지침에 따라 웹훅을 생성하세요:

1. W&B 앱 UI로 이동합니다.
2. **팀 설정**을 클릭합니다.
3. 페이지를 아래로 스크롤하여 **팀 시크릿** 섹션을 찾습니다.
4. **새 시크릿** 버튼을 클릭합니다.
5. 모달이 나타납니다. 시크릿의 이름을 **시크릿 이름** 필드에 입력합니다.
6. **시크릿** 필드에 시크릿을 추가합니다.
7. (선택 사항) 웹훅이 시크릿 키나 토큰을 추가로 필요로 한다면, 엑세스 토큰과 같은 다른 시크릿을 생성하기 위해 5번과 6번 단계를 반복합니다.

웹훅을 구성할 때 웹훅 자동화에 사용할 시크릿을 지정하세요. 자세한 정보는 [웹훅 구성](#configure-a-webhook) 섹션을 참조하세요.

:::tip
시크릿을 생성하면 W&B 워크플로우에서 `$`를 사용하여 해당 시크릿에 엑세스할 수 있습니다.
:::

### 웹훅 구성
웹훅을 사용하려면 먼저 W&B 앱 UI에서 해당 웹훅을 구성해야 합니다.

:::info
* W&B 팀을 위해 웹훅을 구성할 수 있는 사람은 W&B 관리자뿐입니다.
* 웹훅이 시크릿 키나 토큰을 추가로 필요로 한다면 이미 [하나 이상의 시크릿을 생성](#add-a-secret-for-authentication-or-authorization)한 상태여야 합니다.
:::

1. W&B 앱 UI로 이동합니다.
2. **팀 설정**을 클릭합니다.
3. **웹훅** 섹션을 찾을 때까지 페이지를 아래로 스크롤합니다.
4. **새 웹훅** 버튼을 클릭합니다.
5. **이름** 필드에 웹훅의 이름을 입력합니다.
6. **URL** 필드에 웹훅의 엔드포인트 URL을 입력합니다.
7. (선택 사항) **시크릿** 드롭다운 메뉴에서 웹훅 페이로드를 인증하는 데 사용할 시크릿을 선택합니다.
8. (선택 사항) **엑세스 토큰** 드롭다운 메뉴에서 발신자를 승인하는 데 사용할 엑세스 토큰을 선택합니다.

:::note
웹훅 페이로드의 시크릿과 엑세스 토큰이 POST 요청에서 어디에 지정되어 있는지 보려면 [웹훅 문제 해결](#troubleshoot-your-webhook) 섹션을 참조하세요.
:::

### 웹훅 추가
웹훅을 구성하고 (선택 사항으로) 시크릿을 설정한 후, 프로젝트 워크스페이스로 이동하여 왼쪽 사이드바의 **Automations** 탭을 클릭하세요.

1. **이벤트 유형** 드롭다운에서 [이벤트 유형](#event-types)을 선택합니다.
![](/images/artifacts/artifact_webhook_select_event.png)
2. **컬렉션에서 아티팩트의 새로운 버전이 생성됨** 이벤트를 선택했다면, **아티팩트 컬렉션** 드롭다운에서 자동화가 반응할 아티팩트 컬렉션의 이름을 입력합니다.
![](/images/artifacts/webhook_new_version_artifact.png)
3. **액션 유형** 드롭다운에서 **웹훅**을 선택합니다.
4. **다음 단계** 버튼을 클릭합니다.
5. **웹훅** 드롭다운에서 웹훅을 선택합니다.
![](/images/artifacts/artifacts_webhooks_select_from_dropdown.png)
6. (선택 사항) JSON 표현식 편집기에 페이로드를 제공합니다. 일반적인 유스 케이스 예제는 [예제 페이로드](#example-payloads) 섹션을 참조하세요.
7. **다음 단계**를 클릭합니다.
8. **자동화 이름** 필드에 웹훅 자동화의 이름을 입력합니다.
![](/images/artifacts/artifacts_webhook_name_automation.png)
9. (선택 사항) 웹훅에 대한 설명을 제공하세요.
10. **자동화 생성** 버튼을 클릭합니다.

### 예제 페이로드

다음 탭에서는 일반적인 유스 케이스를 기반으로 한 예제 페이로드를 보여줍니다. 예제 내에서 페이로드 파라미터의 조건 오브젝트를 참조하기 위해 다음 키를 사용합니다:
* `${event_type}` 트리거된 액션의 이벤트 유형을 참조합니다.
* `${event_author}` 트리거된 액션을 유발한 사용자를 참조합니다.
* `${artifact_version}` 트리거된 액션의 특정 아티팩트 버전을 참조합니다. 아티팩트 인스턴스로 전달됩니다.
* `${artifact_version_string}` 트리거된 액션의 특정 아티팩트 버전을 참조합니다. 문자열로 전달됩니다.
* `${artifact_collection_name}` 아티팩트 버전이 연결된 아티팩트 컬렉션의 이름을 참조합니다.
* `${project_name}` 트리거된 액션의 변형을 소유한 프로젝트의 이름을 참조합니다.
* `${entity_name}` 트리거된 액션의 변형을 소유한 엔티티의 이름을 참조합니다.

<Tabs
  defaultValue="github"
  values={[
    {label: 'GitHub repository dispatch', value: 'github'},
    {label: 'Microsoft Teams notification', value: 'microsoft'},
    {label: 'Slack notifications', value: 'slack'},
  ]}>
  <TabItem value="github">

:::info
엑세스 토큰이 GHA 워크플로우를 트리거할 필수 권한 세트를 가지고 있는지 확인하세요. 자세한 정보는 [GitHub Docs](https://docs.github.com/en/rest/repos/repos?#create-a-repository-dispatch-event)를 참조하세요.
:::

  W&B에서 GitHub 액션을 트리거하기 위해 리포지토리 디스패치를 보내세요. 예를 들어, 리포지토리 디스패치를 `on` 키의 트리거로 수용하는 워크플로우가 있다고 가정합니다:

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
웹훅 페이로드의 `event_type` 키는 GitHub 워크플로우 YAML 파일의 `types` 필드와 일치해야 합니다.
:::

  자동화가 구성된 이벤트 또는 모델 버전에 따라 렌더링된 템플릿 문자열의 내용 및 위치가 달라집니다. `${event_type}`은 "LINK_ARTIFACT" 또는 "ADD_ARTIFACT_ALIAS"로 렌더링됩니다. 예제 매핑은 다음과 같습니다:

  ```json
  ${event_type} --> "LINK_ARTIFACT" 또는 "ADD_ARTIFACT_ALIAS"
  ${event_author} --> "<wandb-user>"
  ${artifact_version} --> "wandb-artifact://_id/QXJ0aWZhY3Q6NTE3ODg5ODg7""
  ${artifact_version_string} --> "<entity>/<project_name>/<artifact_name>:<alias>"
  ${artifact_collection_name} --> "<artifact_collection_name>"
  ${project_name} --> "<project_name>"
  ${entity_name} --> "<entity>"
  ```

  템플릿 문자열을 사용하여 W&B로부터 GitHub Actions 및 기타 툴에 동적으로 컨텍스트를 전달하세요. 이러한 툴이 Python 스크립트를 호출할 수 있다면, [W&B API](../artifacts/download-and-use-an-artifact.md)를 통해 W&B 아티팩트를 소비할 수 있습니다.

  리포지토리 디스패치에 대한 자세한 내용은 [GitHub Marketplace의 공식 문서](https://github.com/marketplace/actions/repository-dispatch)를 참조하세요.

  </TabItem>
  <TabItem value="microsoft">

  Teams 채널에 대한 웹훅 URL을 얻기 위해 'Incoming Webhook'을 구성하세요. 다음은 예제 페이로드입니다:
  
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
  이 예시에는 실행 시점에 W&B 데이터를 페이로드에 삽입할 수 있도록 템플릿 문자열을 사용할 수 있습니다.

  </TabItem>
  <TabItem value="slack">

  Slack 앱을 설정하고 [Slack API 문서](https://api.slack.com/messaging/webhooks)에서 강조된 지침을 따르며 인커밍 웹훅 인테그레이션을 추가하세요. 'Bot User OAuth 토큰' 아래에 지정된 시크릿이 W&B 웹훅의 엑세스 토큰으로 설정되었는지 확인하세요.
  
  다음은 예제 페이로드입니다:

  ```json
    {
        "text": "New alert from WANDB!",
    "blocks": [
        {
                "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "Artifact event: ${event_type}"
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

W&B 앱 UI 또는 Bash 스크립트를 통해 웹훅을 인터랙티브로 문제 해결하세요. 새로운 웹훅 생성 또는 기존 웹훅 편집 시 웹훅을 문제 해결할 수 있습니다.

<Tabs
  defaultValue="app"
  values={[
    {label: 'W&B App UI', value: 'app'},
    {label: 'Bash script', value: 'bash'},
  ]}>
  <TabItem value="app">

W&B 앱 UI를 사용하여 웹훅을 인터랙티브로 테스트하세요.

1. W&B 팀 설정 페이지로 이동합니다.
2. **웹훅** 섹션으로 스크롤합니다.
3. 웹훅 이름 옆에 있는 가로 세 점(미트볼 아이콘)을 클릭합니다.
4. **테스트**를 선택합니다.
5. 나타나는 UI 패널에서 POST 요청을 필드에 붙여넣습니다.
![](/images/models/webhook_ui.png)
6. **웹훅 테스트**를 클릭합니다.

W&B 앱 UI에서는 엔드포인트에 의해 수행된 응답을 표시합니다.

![](/images/models/webhook_ui_testing.gif)

실제 사례를 보려면 [Weights & Biases에서 웹훅 테스트](https://www.youtube.com/watch?v=bl44fDpMGJw&ab_channel=Weights%26Biases) 유튜브 비디오를 참조하세요.

  </TabItem>
  <TabItem value="bash">

다음 bash 스크립트는 웹훅 자동화가 트리거될 때 W&B가 웹훅으로 보내는 POST 요청과 유사한 POST 요청을 생성합니다.

아래의 코드를 셸 스크립트에 복사하여 붙여넣어 웹훅을 문제 해결하세요. 다음에 대한 자신의 값을 지정하세요:

* `ACCESS_TOKEN`
* `SECRET`
* `PAYLOAD`
* `API_ENDPOINT`

```sh title="webhook_test.sh"
#!/bin/bash

# Your access token and secret
ACCESS_TOKEN="your_api_key" 
SECRET="your_api_secret"

# 보내고자 하는 데이터 (예: JSON 형식)
PAYLOAD='{"key1": "value1", "key2": "value2"}'

# HMAC 서명 생성
# 보안을 위해, Wandb는 HMAC와 SHA-256 알고리즘을 사용하여 
# 웹훅과 관련된 공유 시크릿 키 및 페이로드에서 계산된 헤더에 X-Wandb-Signature를 포함합니다.
SIGNATURE=$(echo -n "$PAYLOAD" | openssl dgst -sha256 -hmac "$SECRET" -binary | base64)

# cURL 요청 생성
curl -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "X-Wandb-Signature: $SIGNATURE" \
  -d "$PAYLOAD" API_ENDPOINT
```

  </TabItem>
</Tabs>

## Launch 자동화 생성
W&B Job을 자동으로 시작합니다.

:::info
이 섹션은 이미 job, 큐를 생성하고 활성 에이전트가 폴링 중이라고 가정합니다. 자세한 정보는 [W&B Launch 문서](../launch/intro.md)를 참조하세요.
:::

1. **이벤트 유형** 드롭다운에서 이벤트 유형을 선택합니다. 지원되는 이벤트에 대한 정보는 [이벤트 유형](#event-types) 섹션을 참조하세요.
2. (선택 사항) **컬렉션에서 아티팩트의 새로운 버전이 생성됨** 이벤트를 선택했다면, **아티팩트 컬렉션** 드롭다운에서 아티팩트 컬렉션의 이름을 입력합니다.
3. **액션 유형** 드롭다운에서 **Jobs**를 선택합니다.
4. **다음 단계**를 클릭합니다.
5. **Job** 드롭다운에서 W&B Launch job을 선택합니다.
6. **Job 버전** 드롭다운에서 버전을 선택합니다.
7. (선택 사항) 새 job에 대한 하이퍼파라미터 오버라이드를 제공합니다.
8. **대상 프로젝트** 드롭다운에서 프로젝트를 선택합니다.
9. 큐에 job을 대기시키기 위한 큐를 선택합니다.
10. **다음 단계**를 클릭합니다.
11. **자동화 이름** 필드에 webhook 자동화의 이름을 입력합니다.
12. (선택 사항) 웹훅에 대한 설명을 제공합니다.
13. **자동화 생성** 버튼을 클릭합니다.

## 자동화 보기

W&B App UI에서 아티팩트와 연결된 자동화를 봅니다.

1. W&B 앱의 프로젝트 워크스페이스로 이동합니다.
2. 왼쪽 사이드바의 **Automations** 탭을 클릭하세요.

![](/images/artifacts/automations_sidebar.gif)

자동화 섹션 내에서는 프로젝트에서 생성된 각 자동화에 대해 다음 속성을 찾을 수 있습니다:

- **트리거 유형**: 구성된 트리거의 유형.
- **액션 유형**: 자동화를 트리거하는 액션 유형. 사용 가능한 옵션은 Webhooks 및 Launch입니다.
- **액션 이름**: 자동화를 생성할 때 제공된 액션 이름입니다.
- **큐**: job이 대기된 큐의 이름입니다. 웹훅 액션 타입을 선택한 경우 이 필드는 비어 있습니다.

## 자동화 삭제
아티팩트와 연결된 자동화를 삭제합니다. 진행 중인 액션은 액션이 완료되기 전에 자동화가 삭제되더라도 영향을 받지 않습니다.

1. W&B 앱의 프로젝트 워크스페이스로 이동합니다.
2. 왼쪽 사이드바에서 **Automations** 탭을 클릭합니다.
3. 목록에서 보고자하는 자동화의 이름을 선택합니다.
4. 자동화 이름 옆에 마우스를 가져가서 세 점 메뉴(세로로 세 방향)를 클릭합니다.
5. **삭제**를 선택합니다.
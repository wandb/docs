---
title: Model registry automations
description: 자동화를 사용하여 모델 CI (자동화된 모델 평가 파이프라인) 및 모델 배포를 수행합니다.
menu:
  default:
    identifier: ko-guides-models-automations-model-registry-automations
    parent: automations
url: guides/model_registry/model-registry-automations
---

자동화 기능을 생성하여 자동화된 모델 테스트 및 배포와 같은 워크플로우 단계를 트리거합니다. 자동화를 생성하려면 [이벤트 유형]({{< relref path="#event-types" lang="ko" >}})을 기반으로 발생시키려는 동작을 정의합니다.

예를 들어, 등록된 모델의 새 버전을 추가할 때 자동으로 모델을 GitHub에 배포하는 트리거를 생성할 수 있습니다.

{{% alert %}}
자동화 관련 튜토리얼을 찾고 계신가요?
1. [이](https://wandb.ai/wandb/wandb-model-cicd/reports/Model-CI-CD-with-W-B--Vmlldzo0OTcwNDQw) 튜토리얼에서는 모델 평가 및 배포를 위해 Github Action을 트리거하는 자동화를 설정하는 방법을 보여줍니다.
2. [이](https://youtube.com/playlist?list=PLD80i8An1OEGECFPgY-HPCNjXgGu-qGO6&feature=shared) 비디오 시리즈에서는 웹훅 기본 사항과 W&B에서 웹훅을 설정하는 방법을 보여줍니다.
3. [이](https://www.youtube.com/watch?v=s5CMj_w3DaQ) 데모에서는 모델을 Sagemaker Endpoint에 배포하는 자동화를 설정하는 방법을 자세히 설명합니다.
{{% /alert %}}

## 이벤트 유형
*이벤트*는 W&B 에코시스템에서 발생하는 변경 사항입니다. Model Registry는 다음 두 가지 이벤트 유형을 지원합니다.

- 새 모델 후보를 테스트하려면 **등록된 모델에 새 Artifacts 연결**을 사용합니다.
- `deploy`와 같이 워크플로우의 특별한 단계를 나타내는 에일리어스를 지정하고 새 모델 버전에 해당 에일리어스가 적용될 때마다 지정하려면 **등록된 모델 버전에 새 에일리어스 추가**를 사용합니다.

[모델 버전 연결]({{< relref path="/guides/models/registry/link_version.md" lang="ko" >}}) 및 [사용자 지정 에일리어스 생성]({{< relref path="/guides/core/artifacts/create-a-custom-alias.md" lang="ko" >}})을 참조하세요.

## 웹훅 자동화 생성
W&B App UI를 사용하여 동작에 따라 웹훅을 자동화합니다. 이렇게 하려면 먼저 웹훅을 설정한 다음 웹훅 자동화를 구성합니다.

{{% alert %}}
웹훅의 엔드포인트에는 정규화된 도메인 이름이 있어야 합니다. W&B는 IP 어드레스 또는 `localhost`와 같은 호스트 이름으로 엔드포인트에 연결하는 것을 지원하지 않습니다. 이러한 제한은 서버 측 요청 위조(SSRF) 공격 및 기타 관련 위협 요인으로부터 보호하는 데 도움이 됩니다.
{{% /alert %}}

### 인증 또는 권한 부여를 위한 비밀 키 추가
비밀 키는 자격 증명, API 키, 비밀번호, 토큰 등과 같은 개인 문자열을 난독화할 수 있는 팀 수준 변수입니다. W&B는 일반 텍스트 콘텐츠를 보호하려는 문자열을 저장하는 데 비밀 키를 사용하는 것이 좋습니다.

웹훅에서 비밀 키를 사용하려면 먼저 해당 비밀 키를 팀의 비밀 키 관리자에 추가해야 합니다.

{{% alert %}}
* W&B 관리자만 비밀 키를 생성, 편집 또는 삭제할 수 있습니다.
* HTTP POST 요청을 보내는 외부 서버에서 비밀 키를 사용하지 않는 경우 이 섹션을 건너뜁니다.
* Azure, GCP 또는 AWS 배포에서 [W&B Server]({{< relref path="/guides/hosting/" lang="ko" >}})를 사용하는 경우에도 비밀 키를 사용할 수 있습니다. 다른 배포 유형을 사용하는 경우 W&B 계정 팀에 문의하여 W&B에서 비밀 키를 사용하는 방법에 대해 논의하세요.
{{% /alert %}}

웹훅 자동화를 사용할 때 생성하는 것이 좋은 두 가지 유형의 비밀 키가 있습니다.

* **엑세스 토큰**: 보낸 사람에게 권한을 부여하여 웹훅 요청을 안전하게 보호합니다.
* **비밀 키**: 페이로드에서 전송된 데이터의 진위성과 무결성을 보장합니다.

웹훅을 생성하려면 아래 지침을 따르세요.

1. W&B App UI로 이동합니다.
2. **팀 설정**을 클릭합니다.
3. 페이지를 아래로 스크롤하여 **팀 비밀 키** 섹션을 찾습니다.
4. **새 비밀 키** 버튼을 클릭합니다.
5. 모달이 나타납니다. **비밀 키 이름** 필드에 비밀 키 이름을 입력합니다.
6. **비밀 키** 필드에 비밀 키를 추가합니다.
7. (선택 사항) 웹훅에서 웹훅을 인증하는 데 필요한 추가 비밀 키 또는 토큰이 필요한 경우 5단계와 6단계를 반복하여 다른 비밀 키(예: 엑세스 토큰)를 생성합니다.

웹훅을 구성할 때 웹훅 자동화에 사용할 비밀 키를 지정합니다. 자세한 내용은 [웹훅 구성]({{< relref path="#configure-a-webhook" lang="ko" >}}) 섹션을 참조하세요.

{{% alert %}}
비밀 키를 생성한 후에는 `$`를 사용하여 W&B 워크플로우에서 해당 비밀 키에 엑세스할 수 있습니다.
{{% /alert %}}

{{% alert color="secondary" %}}
W&B Server에서 비밀 키를 사용하는 경우 보안 요구 사항을 충족하는 보안 조치를 구성할 책임은 사용자에게 있습니다.

AWS, GCP 또는 Azure에서 제공하는 클라우드 비밀 키 관리자의 W&B 인스턴스에 비밀 키를 저장하는 것이 좋습니다. AWS, GCP 및 Azure에서 제공하는 비밀 키 관리자는 고급 보안 기능으로 구성됩니다.

Kubernetes 클러스터를 비밀 키 저장소의 백엔드로 사용하는 것은 권장하지 않습니다. 클라우드 비밀 키 관리자(AWS, GCP 또는 Azure)의 W&B 인스턴스를 사용할 수 없고 클러스터를 사용하는 경우 발생할 수 있는 보안 취약점을 방지하는 방법을 이해하는 경우에만 Kubernetes 클러스터를 고려하세요.
{{% /alert %}}

### 웹훅 구성
웹훅을 사용하기 전에 먼저 W&B App UI에서 해당 웹훅을 구성합니다.

{{% alert %}}
* W&B 관리자만 W&B Team에 대한 웹훅을 구성할 수 있습니다.
* 웹훅에서 웹훅을 인증하는 데 필요한 추가 비밀 키 또는 토큰이 필요한 경우 [하나 이상의 비밀 키를 이미 생성했는지]({{< relref path="#add-a-secret-for-authentication-or-authorization" lang="ko" >}}) 확인하세요.
{{% /alert %}}

1. W&B App UI로 이동합니다.
2. **팀 설정**을 클릭합니다.
4. 페이지를 아래로 스크롤하여 **웹훅** 섹션을 찾습니다.
5. **새 웹훅** 버튼을 클릭합니다.
6. **이름** 필드에 웹훅 이름을 입력합니다.
7. **URL** 필드에 웹훅에 대한 엔드포인트 URL을 입력합니다.
8. (선택 사항) **비밀 키** 드롭다운 메뉴에서 웹훅 페이로드를 인증하는 데 사용할 비밀 키를 선택합니다.
9. (선택 사항) **엑세스 토큰** 드롭다운 메뉴에서 보낸 사람에게 권한을 부여하는 데 사용할 엑세스 토큰을 선택합니다.
9. (선택 사항) **엑세스 토큰** 드롭다운 메뉴에서 웹훅을 인증하는 데 필요한 추가 비밀 키 또는 토큰(예: 엑세스 토큰)을 선택합니다.

{{% alert %}}
POST 요청에서 비밀 키와 엑세스 토큰이 지정된 위치를 보려면 [웹훅 문제 해결]({{< relref path="#troubleshoot-your-webhook" lang="ko" >}}) 섹션을 참조하세요.
{{% /alert %}}

### 웹훅 추가
웹훅을 구성하고 (선택 사항) 비밀 키를 구성했으면 [https://wandb.ai/registry/model](https://wandb.ai/registry/model)에서 Model Registry App으로 이동합니다.

1. **이벤트 유형** 드롭다운에서 [이벤트 유형]({{< relref path="#event-types" lang="ko" >}})을 선택합니다.
{{< img src="/images/models/webhook_select_event.png" alt="" >}}
2. (선택 사항) **등록된 모델에 새 버전이 추가됨** 이벤트를 선택한 경우 **등록된 모델** 드롭다운에서 등록된 모델 이름을 입력합니다.
{{< img src="/images/models/webhook_new_version_reg_model.png" alt="" >}}
3. **동작 유형** 드롭다운에서 **웹훅**을 선택합니다.
4. **다음 단계** 버튼을 클릭합니다.
5. **웹훅** 드롭다운에서 웹훅을 선택합니다.
{{< img src="/images/models/webhooks_select_from_dropdown.png" alt="" >}}
6. (선택 사항) JSON 표현식 편집기에서 페이로드를 제공합니다. 일반적인 유스 케이스 예제는 [페이로드 예제]({{< relref path="#example-payloads" lang="ko" >}}) 섹션을 참조하세요.
7. **다음 단계**를 클릭합니다.
8. **자동화 이름** 필드에 웹훅 자동화 이름을 입력합니다.
{{< img src="/images/models/webhook_name_automation.png" alt="" >}}
9. (선택 사항) 웹훅에 대한 설명을 제공합니다.
10. **자동화 생성** 버튼을 클릭합니다.

### 페이로드 예제

다음 탭에서는 일반적인 유스 케이스를 기반으로 한 페이로드 예제를 보여줍니다. 예제 내에서 페이로드 파라미터의 조건 오브젝트를 참조하기 위해 다음 키를 참조합니다.
* `${event_type}` 동작을 트리거한 이벤트 유형을 나타냅니다.
* `${event_author}` 동작을 트리거한 사용자를 나타냅니다.
* `${artifact_version}` 동작을 트리거한 특정 Artifacts 버전을 나타냅니다. Artifacts 인스턴스로 전달됩니다.
* `${artifact_version_string}` 동작을 트리거한 특정 Artifacts 버전을 나타냅니다. 문자열로 전달됩니다.
* `${artifact_collection_name}` Artifacts 버전이 연결된 Artifacts 컬렉션의 이름을 나타냅니다.
* `${project_name}` 동작을 트리거한 변경을 소유하는 프로젝트의 이름을 나타냅니다.
* `${entity_name}` 동작을 트리거한 변경을 소유하는 엔터티의 이름을 나타냅니다.

{{< tabpane text=true >}}
{{% tab header="GitHub repository dispatch" value="github" %}}

{{% alert %}}
엑세스 토큰에 GHA 워크플로우를 트리거하는 데 필요한 권한 집합이 있는지 확인합니다. 자세한 내용은 [이 GitHub 문서](https://docs.github.com/en/rest/repos/repos?#create-a-repository-dispatch-event)를 참조하세요.
{{% /alert %}}
  
  W&B에서 리포지토리 디스패치를 보내 GitHub 액션을 트리거합니다. 예를 들어, `on` 키에 대한 트리거로 리포지토리 디스패치를 허용하는 워크플로우가 있다고 가정합니다.

  ```yaml
  on:
  repository_dispatch:
    types: BUILD_AND_DEPLOY
  ```

  리포지토리의 페이로드는 다음과 같습니다.

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

  {{% alert %}}
  웹훅 페이로드의 `event_type` 키는 GitHub 워크플로우 YAML 파일의 `types` 필드와 일치해야 합니다.
  {{% /alert %}}

  렌더링된 템플릿 문자열의 내용과 위치는 자동화가 구성된 이벤트 또는 모델 버전에 따라 달라집니다. `${event_type}`은 `LINK_ARTIFACT` 또는 `ADD_ARTIFACT_ALIAS`로 렌더링됩니다. 아래에서 예제 매핑을 참조하세요.

  ```json
  ${event_type} --> "LINK_ARTIFACT" 또는 "ADD_ARTIFACT_ALIAS"
  ${event_author} --> "<wandb-user>"
  ${artifact_version} --> "wandb-artifact://_id/QXJ0aWZhY3Q6NTE3ODg5ODg3""
  ${artifact_version_string} --> "<entity>/model-registry/<registered_model_name>:<alias>"
  ${artifact_collection_name} --> "<registered_model_name>"
  ${project_name} --> "model-registry"
  ${entity_name} --> "<entity>"
  ```

  템플릿 문자열을 사용하여 W&B에서 GitHub Actions 및 기타 툴로 컨텍스트를 동적으로 전달합니다. 해당 툴에서 Python 스크립트를 호출할 수 있는 경우 [W&B API]({{< relref path="/guides/core/artifacts/download-and-use-an-artifact.md" lang="ko" >}})를 통해 등록된 모델 Artifacts를 사용할 수 있습니다.

  리포지토리 디스패치에 대한 자세한 내용은 [GitHub Marketplace의 공식 문서](https://github.com/marketplace/actions/repository-dispatch)를 참조하세요.

  모델 평가 및 배포를 위한 자동화를 만드는 방법을 안내하는 [모델 평가를 위한 웹훅 자동화](https://www.youtube.com/watch?v=7j-Mtbo-E74&ab_channel=Weights%26Biases) 및 [모델 배포를 위한 웹훅 자동화](https://www.youtube.com/watch?v=g5UiAFjM2nA&ab_channel=Weights%26Biases) 비디오를 시청하세요.

  Model CI에 Github Actions 웹훅 자동화를 사용하는 방법을 보여주는 W&B [리포트](https://wandb.ai/wandb/wandb-model-cicd/reports/Model-CI-CD-with-W-B--Vmlldzo0OTcwNDQw)를 검토하세요. Modal Labs 웹훅으로 모델 CI를 만드는 방법을 알아보려면 [이 GitHub 리포지토리](https://github.com/hamelsmu/wandb-modal-webhook)를 확인하세요.

{{% /tab %}}

{{% tab header="Microsoft Teams notification" value="microsoft"%}}

  구성하여 Teams Channel에 대한 웹훅 URL을 가져오려면 'Incoming Webhook'을 구성합니다. 다음은 페이로드 예제입니다.
  
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

  템플릿 문자열을 사용하여 실행 시점에 W&B 데이터를 페이로드에 삽입할 수 있습니다(위의 Teams 예제에서와 같이).

{{% /tab %}}

{{% tab header="Slack notifications" value="slack"%}}

  [Slack API 문서](https://api.slack.com/messaging/webhooks)에 강조 표시된 지침에 따라 Slack 앱을 설정하고 들어오는 웹훅 인테그레이션을 추가합니다. `Bot User OAuth Token` 아래에 지정된 비밀 키가 W&B 웹훅의 엑세스 토큰인지 확인합니다.
  
  다음은 페이로드 예제입니다.

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

{{% /tab %}}
{{< /tabpane >}}

### 웹훅 문제 해결

W&B App UI를 사용하여 대화식으로 또는 Bash 스크립트를 사용하여 프로그래밍 방식으로 웹훅 문제를 해결합니다. 새 웹훅을 만들거나 기존 웹훅을 편집할 때 웹훅 문제를 해결할 수 있습니다.

{{< tabpane text=true >}}
{{% tab header="W&B App UI" value="app" %}}

  W&B App UI를 사용하여 대화식으로 웹훅을 테스트합니다.

  1. W&B Team 설정 페이지로 이동합니다.
  2. **웹훅** 섹션으로 스크롤합니다.
  3. 웹훅 이름 옆에 있는 가로 세 개의 점(미트볼 아이콘)을 클릭합니다.
  4. **테스트**를 선택합니다.
  5. 나타나는 UI 패널에서 나타나는 필드에 POST 요청을 붙여넣습니다.
     {{< img src="/images/models/webhook_ui.png" >}}
  6. **웹훅 테스트**를 클릭합니다.

  W&B App UI 내에서 W&B는 엔드포인트에서 수행한 응답을 게시합니다.

  {{< img src="/images/models/webhook_ui_testing.gif" alt="" >}}

  실제 예제는 [Weights & Biases에서 웹훅 테스트](https://www.youtube.com/watch?v=bl44fDpMGJw&ab_channel=Weights%26Biases) 비디오를 시청하세요.

{{% /tab %}}

{{% tab header="Bash script" value="bash" %}}

  다음 Bash 스크립트는 트리거될 때 W&B가 웹훅 자동화에 보내는 POST 요청과 유사한 POST 요청을 생성합니다.

  아래 코드를 복사하여 셸 스크립트에 붙여넣어 웹훅 문제를 해결합니다. 다음 값에 대한 고유한 값을 지정합니다.

  * `ACCESS_TOKEN`
  * `SECRET`
  * `PAYLOAD`
  * `API_ENDPOINT`

  ```sh { title = "webhook_test.sh" }
  #!/bin/bash

  # Your access token and secret
  ACCESS_TOKEN="your_api_key" 
  SECRET="your_api_secret"

  # The data you want to send (for example, in JSON format)
  PAYLOAD='{"key1": "value1", "key2": "value2"}'

  # Generate the HMAC signature
  # For security, Wandb includes the X-Wandb-Signature in the header computed 
  # from the payload and the shared secret key associated with the webhook 
  # using the HMAC with SHA-256 algorithm.
  SIGNATURE=$(echo -n "$PAYLOAD" | openssl dgst -sha256 -hmac "$SECRET" -binary | base64)

  # Make the cURL request
  curl -X POST \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $ACCESS_TOKEN" \
    -H "X-Wandb-Signature: $SIGNATURE" \
    -d "$PAYLOAD" API_ENDPOINT
  ```

{{% /tab %}}
{{< /tabpane >}}

## 자동화 보기

W&B App UI에서 등록된 모델과 연결된 자동화를 봅니다.

1. [https://wandb.ai/registry/model](https://wandb.ai/registry/model)에서 Model Registry App으로 이동합니다.
2. 등록된 모델을 선택합니다.
3. 페이지 하단으로 스크롤하여 **자동화** 섹션으로 이동합니다.

자동화 섹션 내에서 선택한 모델에 대해 생성된 자동화의 다음 속성을 찾을 수 있습니다.

- **트리거 유형**: 구성된 트리거 유형입니다.
- **동작 유형**: 자동화를 트리거하는 동작 유형입니다.
- **동작 이름**: 자동화를 생성할 때 제공한 동작 이름입니다.
- **대기열**: 작업을 대기열에 추가한 대기열의 이름입니다. 이 필드는 웹훅 동작 유형을 선택한 경우 비워 둡니다.

## 자동화 삭제
모델과 연결된 자동화를 삭제합니다. 동작이 완료되기 전에 해당 자동화를 삭제하면 진행 중인 동작은 영향을 받지 않습니다.

1. [https://wandb.ai/registry/model](https://wandb.ai/registry/model)에서 Model Registry App으로 이동합니다.
2. 등록된 모델을 클릭합니다.
3. 페이지 하단으로 스크롤하여 **자동화** 섹션으로 이동합니다.
4. 자동화 이름 옆으로 마우스를 가져간 다음 케밥(세 개의 세로 점) 메뉴를 클릭합니다.
5. **삭제**를 선택합니다.

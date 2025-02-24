---
title: Trigger CI/CD events when artifact changes
description: 프로젝트에서 프로젝트 범위의 아티팩트 자동화를 사용하여 아티팩트 컬렉션에서 에일리어스 또는 버전이 생성되거나 변경될 때 작업을
  트리거하세요.
menu:
  default:
    identifier: ko-guides-models-automations-project-scoped-automations
    parent: automations
url: guides/artifacts/project-scoped-automations
---

Artifact가 변경될 때 트리거되는 자동화를 생성합니다. Artifact 버전 관리를 위한 다운스트림 작업을 자동화하려면 Artifact 자동화를 사용하세요. 자동화를 생성하려면 [이벤트 유형]({{< relref path="#event-types" lang="ko" >}})에 따라 발생할 작업을 정의합니다.

{{% alert %}}
Artifact 자동화는 프로젝트 범위로 지정됩니다. 즉, 프로젝트 내의 이벤트만 Artifact 자동화를 트리거합니다.

이는 W&B 모델 레지스트리에서 생성된 자동화와 대조됩니다. 모델 레지스트리에서 생성된 자동화는 모델 레지스트리 범위 내에 있습니다. 이는 [모델 레지스트리]({{< relref path="/guides/models/registry/model_registry/" lang="ko" >}})에 연결된 모델 버전에 대해 이벤트가 수행될 때 트리거됩니다. 모델 버전에 대한 자동화를 생성하는 방법에 대한 자세한 내용은 [모델 CI/CD 자동화]({{< relref path="/guides/models/automations/model-registry-automations.md" lang="ko" >}}) 페이지의 [모델 레지스트리 챕터]({{< relref path="/guides/models/registry/model_registry/" lang="ko" >}})를 참조하세요.
{{% /alert %}}

## 이벤트 유형
*이벤트*는 W&B 에코시스템에서 발생하는 변경 사항입니다. 프로젝트에서 Artifact 컬렉션에 대해 두 가지 다른 이벤트 유형( **Artifact의 새 버전이 컬렉션에 생성됨** 및 **Artifact 에일리어스가 추가됨**)을 정의할 수 있습니다.

{{% alert %}}
Artifact의 각 버전에 반복 작업을 적용하려면 **Artifact의 새 버전이 컬렉션에 생성됨** 이벤트 유형을 사용하세요. 예를 들어, 새로운 데이터셋 Artifact 버전이 생성되면 자동으로 트레이닝 작업을 시작하는 자동화를 생성할 수 있습니다.

특정 에일리어스가 Artifact 버전에 적용될 때 활성화되는 자동화를 생성하려면 **Artifact 에일리어스가 추가됨** 이벤트 유형을 사용하세요. 예를 들어, 누군가가 "test-set-quality-check" 에일리어스를 Artifact에 추가할 때 작업을 트리거하는 자동화를 생성하여 해당 데이터셋에 대한 다운스트림 처리를 트리거할 수 있습니다.
{{% /alert %}}

## Webhook 자동화 생성
W&B App UI를 사용하여 작업을 기반으로 webhook을 자동화합니다. 이를 위해 먼저 webhook을 설정한 다음 webhook 자동화를 구성합니다.

{{% alert %}}
어드레스 레코드(A 레코드)가 있는 webhook에 대한 엔드포인트를 지정합니다. W&B는 `[0-255].[0-255].[0-255].[0.255]`와 같은 IP 어드레스로 직접 노출되는 엔드포인트 또는 `localhost`로 노출되는 엔드포인트에 대한 연결을 지원하지 않습니다. 이 제한은 서버 측 요청 위조(SSRF) 공격 및 기타 관련 위협 벡터로부터 보호하는 데 도움이 됩니다.
{{% /alert %}}

### 인증 또는 권한 부여를 위한 보안 비밀 추가
보안 비밀은 자격 증명, API 키, 비밀번호, 토큰 등과 같은 개인 문자열을 난독화할 수 있는 팀 수준 변수입니다. 일반 텍스트 콘텐츠를 보호하려는 문자열을 저장하려면 보안 비밀을 사용하는 것이 좋습니다.

webhook에서 보안 비밀을 사용하려면 먼저 해당 보안 비밀을 팀의 보안 비밀 관리자에 추가해야 합니다.

{{% alert %}}
* W&B 관리자만 보안 비밀을 생성, 편집 또는 삭제할 수 있습니다.
* HTTP POST 요청을 보내는 외부 서버가 보안 비밀을 사용하지 않는 경우 이 섹션을 건너뛰세요.
* Azure, GCP 또는 AWS 배포에서 [W&B 서버]({{< relref path="/guides/hosting/" lang="ko" >}})를 사용하는 경우에도 보안 비밀을 사용할 수 있습니다. 다른 배포 유형을 사용하는 경우 W&B 계정 팀에 문의하여 W&B에서 보안 비밀을 사용하는 방법에 대해 논의하세요.
{{% /alert %}}

webhook 자동화를 사용할 때 생성하는 것이 좋은 두 가지 유형의 보안 비밀은 다음과 같습니다.

* **액세스 토큰**: 발신자가 webhook 요청을 보호하도록 권한을 부여합니다.
* **보안 비밀**: 페이로드에서 전송된 데이터의 진위성과 무결성을 보장합니다.

다음 지침에 따라 webhook을 생성합니다.

1. W&B App UI로 이동합니다.
2. **팀 설정**을 클릭합니다.
3. **팀 보안 비밀** 섹션이 나올 때까지 페이지를 아래로 스크롤합니다.
4. **새 보안 비밀** 버튼을 클릭합니다.
5. 모달이 나타납니다. **보안 비밀 이름** 필드에 보안 비밀 이름을 입력합니다.
6. **보안 비밀** 필드에 보안 비밀을 추가합니다.
7. (선택 사항) webhook 인증에 추가 보안 키 또는 토큰이 필요한 경우 5단계와 6단계를 반복하여 다른 보안 비밀(예: 액세스 토큰)을 생성합니다.

webhook을 구성할 때 webhook 자동화에 사용할 보안 비밀을 지정합니다. 자세한 내용은 [Webhook 구성]({{< relref path="#configure-a-webhook" lang="ko" >}}) 섹션을 참조하세요.

{{% alert %}}
보안 비밀을 생성하면 `$`를 사용하여 W&B 워크플로우에서 해당 보안 비밀에 액세스할 수 있습니다.
{{% /alert %}}

### Webhook 구성
webhook을 사용하기 전에 먼저 W&B App UI에서 해당 webhook을 구성해야 합니다.

{{% alert %}}
* W&B 관리자만 W&B 팀에 대한 webhook을 구성할 수 있습니다.
* webhook 인증에 추가 보안 키 또는 토큰이 필요한 경우 [하나 이상의 보안 비밀을 이미 생성했는지]({{< relref path="#add-a-secret-for-authentication-or-authorization" lang="ko" >}}) 확인하세요.
{{% /alert %}}

1. W&B App UI로 이동합니다.
2. **팀 설정**을 클릭합니다.
3. **Webhooks** 섹션이 나올 때까지 페이지를 아래로 스크롤합니다.
4. **새 webhook** 버튼을 클릭합니다.
5. **이름** 필드에 webhook 이름을 입력합니다.
6. **URL** 필드에 webhook에 대한 엔드포인트 URL을 입력합니다.
7. (선택 사항) **보안 비밀** 드롭다운 메뉴에서 webhook 페이로드를 인증하는 데 사용할 보안 비밀을 선택합니다.
8. (선택 사항) **액세스 토큰** 드롭다운 메뉴에서 발신자를 인증하는 데 사용할 액세스 토큰을 선택합니다.
9. (선택 사항) **액세스 토큰** 드롭다운 메뉴에서 webhook을 인증하는 데 필요한 추가 보안 키 또는 토큰(예: 액세스 토큰)을 선택합니다.

{{% alert %}}
POST 요청에서 보안 비밀과 액세스 토큰이 지정된 위치를 보려면 [Webhook 문제 해결]({{< relref path="#troubleshoot-your-webhook" lang="ko" >}}) 섹션을 참조하세요.
{{% /alert %}}

### Webhook 추가
webhook이 구성되고 (선택적으로) 보안 비밀이 있으면 프로젝트 워크스페이스로 이동합니다. 왼쪽 사이드바에서 **자동화** 탭을 클릭합니다.

1. **이벤트 유형** 드롭다운에서 [이벤트 유형]({{< relref path="#event-types" lang="ko" >}})을 선택합니다.
{{< img src="/images/artifacts/artifact_webhook_select_event.png" alt="" >}}
2. **Artifact의 새 버전이 컬렉션에 생성됨** 이벤트를 선택한 경우 **Artifact 컬렉션** 드롭다운에서 자동화가 응답해야 하는 Artifact 컬렉션의 이름을 입력합니다.
{{< img src="/images/artifacts/webhook_new_version_artifact.png" alt="" >}}
3. **액션 유형** 드롭다운에서 **Webhooks**를 선택합니다.
4. **다음 단계** 버튼을 클릭합니다.
5. **Webhook** 드롭다운에서 webhook을 선택합니다.
{{< img src="/images/artifacts/artifacts_webhooks_select_from_dropdown.png" alt="" >}}
6. (선택 사항) JSON 표현식 편집기에서 페이로드를 입력합니다. 일반적인 유스 케이스 예제는 [페이로드 예제]({{< relref path="#example-payloads" lang="ko" >}}) 섹션을 참조하세요.
7. **다음 단계**를 클릭합니다.
8. **자동화 이름** 필드에 webhook 자동화 이름을 입력합니다.
{{< img src="/images/artifacts/artifacts_webhook_name_automation.png" alt="" >}}
9. (선택 사항) webhook에 대한 설명을 입력합니다.
10. **자동화 생성** 버튼을 클릭합니다.

### 페이로드 예제

다음 탭에서는 일반적인 유스 케이스를 기반으로 한 페이로드 예제를 보여줍니다. 예제 내에서 페이로드 파라미터의 조건 오브젝트를 참조하기 위해 다음 키를 참조합니다.
* `${event_type}` 액션을 트리거한 이벤트 유형을 참조합니다.
* `${event_author}` 액션을 트리거한 사용자를 참조합니다.
* `${artifact_version}` 액션을 트리거한 특정 Artifact 버전을 참조합니다. Artifact 인스턴스로 전달됩니다.
* `${artifact_version_string}` 액션을 트리거한 특정 Artifact 버전을 참조합니다. 문자열로 전달됩니다.
* `${artifact_collection_name}` Artifact 버전이 연결된 Artifact 컬렉션의 이름을 참조합니다.
* `${project_name}` 액션을 트리거한 변경 사항을 소유한 프로젝트 이름을 참조합니다.
* `${entity_name}` 액션을 트리거한 변경 사항을 소유한 엔티티 이름을 참조합니다.

{{< tabpane text=true >}}

{{% tab header="GitHub repository dispatch" value="github" %}}
{{% alert %}}
액세스 토큰에 GHA 워크플로우를 트리거하는 데 필요한 권한 세트가 있는지 확인하세요. 자세한 내용은 [이 GitHub 문서를 참조하세요](https://docs.github.com/en/rest/repos/repos?#create-a-repository-dispatch-event).
{{% /alert %}}

W&B에서 리포지토리 디스패치를 보내 GitHub 액션을 트리거합니다. 예를 들어 `on` 키에 대한 트리거로 리포지토리 디스패치를 허용하는 워크플로우가 있다고 가정합니다.

```yaml
on:
  repository_dispatch:
    types: BUILD_AND_DEPLOY
```

리포지토리에 대한 페이로드는 다음과 같습니다.

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
webhook 페이로드의 `event_type` 키는 GitHub 워크플로우 YAML 파일의 `types` 필드와 일치해야 합니다.
{{% /alert %}}

렌더링된 템플릿 문자열의 내용과 위치는 자동화가 구성된 이벤트 또는 모델 버전에 따라 다릅니다. `${event_type}`은 `LINK_ARTIFACT` 또는 `ADD_ARTIFACT_ALIAS`로 렌더링됩니다. 아래 매핑 예제를 참조하세요.

```json
${event_type} --> "LINK_ARTIFACT" 또는 "ADD_ARTIFACT_ALIAS"
${event_author} --> "<wandb-user>"
${artifact_version} --> "wandb-artifact://_id/QXJ0aWZhY3Q6NTE3ODg5ODg3""
${artifact_version_string} --> "<entity>/<project_name>/<artifact_name>:<alias>"
${artifact_collection_name} --> "<artifact_collection_name>"
${project_name} --> "<project_name>"
${entity_name} --> "<entity>"
```

템플릿 문자열을 사용하여 W&B에서 GitHub 액션 및 기타 툴로 컨텍스트를 동적으로 전달합니다. 이러한 툴이 Python 스크립트를 호출할 수 있는 경우 [W&B API]({{< relref path="/guides/core/artifacts/download-and-use-an-artifact.md" lang="ko" >}})를 통해 W&B Artifact를 사용할 수 있습니다.

리포지토리 디스패치에 대한 자세한 내용은 [GitHub 마켓플레이스의 공식 문서](https://github.com/marketplace/actions/repository-dispatch)를 참조하세요.
{{% /tab %}}

{{% tab header="Microsoft Teams 알림" value="microsoft"%}}

구성하여 Teams 채널에 대한 webhook URL을 얻으려면 ‘Incoming Webhook'을 구성하세요. 다음은 페이로드 예제입니다.
  
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

템플릿 문자열을 사용하여 실행 시 W&B 데이터를 페이로드에 삽입할 수 있습니다(위의 Teams 예제 참조).

{{% /tab %}}

{{% tab header="Slack 알림" value="slack"%}}

[Slack API 문서](https://api.slack.com/messaging/webhooks)에 강조 표시된 지침에 따라 Slack 앱을 설정하고 들어오는 webhook 통합을 추가합니다. `Bot User OAuth Token` 아래에 지정된 보안 비밀이 W&B webhook의 액세스 토큰인지 확인합니다.
  
다음은 페이로드 예제입니다.

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
{{% /tab %}}

{{< /tabpane >}}

### Webhook 문제 해결

W&B App UI를 사용하여 대화형으로 또는 Bash 스크립트를 사용하여 프로그래밍 방식으로 webhook 문제를 해결합니다. 새 webhook을 만들거나 기존 webhook을 편집할 때 webhook 문제를 해결할 수 있습니다.

{{< tabpane text=true >}}
{{% tab header="W&B App UI" value="app" %}}

W&B App UI를 사용하여 webhook을 대화형으로 테스트합니다.

1. W&B 팀 설정 페이지로 이동합니다.
2. **Webhooks** 섹션으로 스크롤합니다.
3. webhook 이름 옆에 있는 가로 세 개의 점(미트볼 아이콘)을 클릭합니다.
4. **테스트**를 선택합니다.
5. 나타나는 UI 패널에서 POST 요청을 나타나는 필드에 붙여넣습니다.
{{< img src="/images/models/webhook_ui.png" alt="" >}}
6. **Webhook 테스트**를 클릭합니다.

W&B App UI 내에서 W&B는 엔드포인트에서 수행한 응답을 게시합니다.

{{< img src="/images/models/webhook_ui_testing.gif" alt="" >}}

실제 예제를 보려면 [Weights & Biases에서 Webhook 테스트](https://www.youtube.com/watch?v=bl44fDpMGJw&ab_channel=Weights%26Biases) YouTube 비디오를 참조하세요.
{{% /tab %}}

{{% tab header="Bash 스크립트" value="bash"%}}

다음 bash 스크립트는 트리거될 때 W&B가 webhook 자동화에 보내는 POST 요청과 유사한 POST 요청을 생성합니다.

아래 코드를 셸 스크립트에 복사하여 붙여넣어 webhook 문제를 해결합니다. 다음 값에 대해 고유한 값을 지정합니다.

* `ACCESS_TOKEN`
* `SECRET`
* `PAYLOAD`
* `API_ENDPOINT`

{{< prism file="/webhook_test.sh" title="webhook_test.sh">}}{{< /prism >}}

{{% /tab %}}
{{< /tabpane >}}

## 자동화 보기

W&B App UI에서 Artifact와 연결된 자동화를 봅니다.

1. W&B App에서 프로젝트 워크스페이스로 이동합니다.
2. 왼쪽 사이드바에서 **자동화** 탭을 클릭합니다.

{{< img src="/images/artifacts/automations_sidebar.gif" alt="" >}}

자동화 섹션 내에서 프로젝트에서 생성된 각 자동화에 대해 다음 속성을 찾을 수 있습니다.

- **트리거 유형**: 구성된 트리거 유형입니다.
- **액션 유형**: 자동화를 트리거하는 액션 유형입니다.
- **액션 이름**: 자동화를 생성할 때 제공한 액션 이름입니다.
- **대기열**: 작업이 대기열에 추가된 대기열의 이름입니다. webhook 액션 유형을 선택한 경우 이 필드는 비워 둡니다.

## 자동화 삭제
Artifact와 연결된 자동화를 삭제합니다. 작업이 완료되기 전에 해당 자동화를 삭제하면 진행 중인 작업은 영향을 받지 않습니다.

1. W&B App에서 프로젝트 워크스페이스로 이동합니다.
2. 왼쪽 사이드바에서 **자동화** 탭을 클릭합니다.
3. 목록에서 보려는 자동화 이름을 선택합니다.
4. 자동화 이름 옆으로 마우스를 가져간 다음 케밥(세 개의 수직 점) 메뉴를 클릭합니다.
5. **삭제**를 선택합니다.

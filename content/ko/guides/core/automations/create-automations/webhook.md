---
title: Create a webhook automation
menu:
  default:
    identifier: ko-guides-core-automations-create-automations-webhook
    parent: automations
weight: 3
---

{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-cloud-only.md" >}}
{{% /pageinfo %}}

이 페이지에서는 webhook [자동화]({{< relref path="/guides/core/automations/" lang="ko" >}})를 만드는 방법을 보여줍니다. Slack 자동화를 만들려면 [Slack 자동화 만들기]({{< relref path="/guides/core/automations/create-automations/slack.md" lang="ko" >}})를 참조하세요.

개략적으로 webhook 자동화를 만들려면 다음 단계를 수행합니다.
1. 필요한 경우 액세스 토큰, 비밀번호 또는 SSH 키와 같이 자동화에 필요한 각 민감한 문자열에 대해 [W&B secret 만들기]({{< relref path="/guides/core/secrets.md" lang="ko" >}})를 수행합니다. secret은 팀 설정에 정의되어 있습니다.
1. [webhook 만들기]({{< relref path="#create-a-webhook" lang="ko" >}})를 수행하여 엔드포인트 및 인증 세부 정보를 정의하고 통합에 필요한 secret에 대한 엑세스 권한을 부여합니다.
1. [자동화 만들기]({{< relref path="#create-an-automation" lang="ko" >}})를 수행하여 감시할 [이벤트]({{< relref path="/guides/core/automations/automation-events.md" lang="ko" >}})와 W&B가 보낼 페이로드를 정의합니다. 페이로드에 필요한 secret에 대한 자동화 엑세스 권한을 부여합니다.

## webhook 만들기
팀 관리자는 팀에 대한 webhook을 추가할 수 있습니다.

{{% alert %}}
webhook에 Bearer 토큰이 필요하거나 페이로드에 민감한 문자열이 필요한 경우 webhook을 만들기 전에 [해당 문자열을 포함하는 secret을 만드세요]({{< relref path="/guides/core/secrets.md#add-a-secret" lang="ko" >}}). webhook에 대해 최대 하나의 엑세스 토큰과 다른 하나의 secret을 구성할 수 있습니다. webhook의 인증 및 권한 부여 요구 사항은 webhook의 서비스에 의해 결정됩니다.
{{% /alert %}}

1. W&B에 로그인하고 팀 설정 페이지로 이동합니다.
2. **Webhooks** 섹션에서 **New webhook**을 클릭합니다.
3. webhook의 이름을 입력합니다.
4. webhook의 엔드포인트 URL을 입력합니다.
5. webhook에 Bearer 토큰이 필요한 경우 **Access token**을 해당 토큰을 포함하는 [secret]({{< relref path="/guides/core/secrets.md" lang="ko" >}})으로 설정합니다. webhook 자동화를 사용할 때 W&B는 `Authorization: Bearer` HTTP 헤더를 엑세스 토큰으로 설정하고 `${ACCESS_TOKEN}` [페이로드 변수]({{< relref path="#payload-variables" lang="ko" >}})에서 토큰에 엑세스할 수 있습니다.
6. webhook의 페이로드에 비밀번호 또는 기타 민감한 문자열이 필요한 경우 **Secret**을 해당 문자열을 포함하는 secret으로 설정합니다. webhook을 사용하는 자동화를 구성할 때 이름 앞에 `$`를 붙여 [페이로드 변수]({{< relref path="#payload-variables" lang="ko" >}})로 secret에 엑세스할 수 있습니다.

    webhook의 엑세스 토큰이 secret에 저장된 경우 secret을 엑세스 토큰으로 지정하려면 _또한_ 다음 단계를 완료해야 합니다.
7. W&B가 엔드포인트에 연결하고 인증할 수 있는지 확인하려면:
    1. 선택적으로 테스트할 페이로드를 제공합니다. 페이로드에서 webhook이 엑세스할 수 있는 secret을 참조하려면 이름 앞에 `$`를 붙입니다. 이 페이로드는 테스트에만 사용되며 저장되지 않습니다. [자동화를 만들 때]({{< relref path="#create-a-webhook-automation" lang="ko" >}}) 자동화의 페이로드를 구성합니다. secret과 엑세스 토큰이 `POST` 요청에 지정된 위치를 보려면 [webhook 문제 해결]({{< relref path="#troubleshoot-your-webhook" lang="ko" >}})을 참조하세요.
    1. **Test**를 클릭합니다. W&B는 구성한 자격 증명을 사용하여 webhook의 엔드포인트에 연결을 시도합니다. 페이로드를 제공한 경우 W&B는 해당 페이로드를 보냅니다.

    테스트가 성공하지 못하면 webhook의 구성을 확인하고 다시 시도하세요. 필요한 경우 [webhook 문제 해결]({{< relref path="#troubleshoot-your-webhook" lang="ko" >}})을 참조하세요.

이제 webhook을 사용하는 [자동화를 만들 수 있습니다]({{< relref path="#create-a-webhook-automation" lang="ko" >}}).

## 자동화 만들기
[webhook을 구성한]({{< relref path="#reate-a-webhook" lang="ko" >}}) 후 **Registry** 또는 **Project**를 선택한 다음 다음 단계에 따라 webhook을 트리거하는 자동화를 만듭니다.

{{< tabpane text=true >}}
{{% tab "Registry" %}}
Registry 관리자는 해당 Registry에서 자동화를 만들 수 있습니다. Registry 자동화는 향후 추가되는 자동화를 포함하여 Registry의 모든 컬렉션에 적용됩니다.

1. W&B에 로그인합니다.
2. Registry 이름을 클릭하여 세부 정보를 확인합니다.
3. Registry로 범위가 지정된 자동화를 만들려면 **Automations** 탭을 클릭한 다음 **Create automation**을 클릭합니다. Registry로 범위가 지정된 자동화는 향후 생성되는 컬렉션을 포함하여 모든 컬렉션에 자동으로 적용됩니다.

    Registry의 특정 컬렉션으로만 범위가 지정된 자동화를 만들려면 컬렉션의 액션 `...` 메뉴를 클릭한 다음 **Create automation**을 클릭합니다. 또는 컬렉션을 보는 동안 컬렉션 세부 정보 페이지의 **Automations** 섹션에 있는 **Create automation** 버튼을 사용하여 컬렉션에 대한 자동화를 만듭니다.
4. 감시할 [**Event**]({{< relref path="/guides/core/automations/automation-events.md" lang="ko" >}})를 선택합니다. 이벤트에 따라 표시되는 추가 필드를 작성합니다. 예를 들어 **An artifact alias is added**를 선택한 경우 **Alias regex**를 지정해야 합니다. **Next step**을 클릭합니다.
5. [webhook]({{< relref path="#create-a-webhook" lang="ko" >}})을 소유한 팀을 선택합니다.
6. **Action type**을 **Webhooks**로 설정한 다음 사용할 [webhook]({{< relref path="#create-a-webhook" lang="ko" >}})을 선택합니다.
7. webhook에 대해 엑세스 토큰을 구성한 경우 `${ACCESS_TOKEN}` [페이로드 변수]({{< relref path="#payload-variables" lang="ko" >}})에서 토큰에 엑세스할 수 있습니다. webhook에 대해 secret을 구성한 경우 이름 앞에 `$`를 붙여 페이로드에서 해당 secret에 엑세스할 수 있습니다. webhook의 요구 사항은 webhook의 서비스에 의해 결정됩니다.
8. **Next step**을 클릭합니다.
9. 자동화 이름을 입력합니다. 선택적으로 설명을 제공합니다. **Create automation**을 클릭합니다.

{{% /tab %}}
{{% tab "Project" %}}
W&B 관리자는 Project에서 자동화를 만들 수 있습니다.

1. W&B에 로그인하고 Project 페이지로 이동합니다.
2. 사이드바에서 **Automations**을 클릭합니다.
3. **Create automation**을 클릭합니다.
4. 감시할 [**Event**]({{< relref path="/guides/core/automations/automation-events.md" lang="ko" >}})를 선택합니다.

    1. 이벤트에 따라 표시되는 추가 필드를 작성합니다. 예를 들어 **An artifact alias is added**를 선택한 경우 **Alias regex**를 지정해야 합니다.

    1. 선택적으로 컬렉션 필터를 지정합니다. 그렇지 않으면 자동화는 향후 추가되는 컬렉션을 포함하여 Project의 모든 컬렉션에 적용됩니다.
    
    **Next step**을 클릭합니다.
5. [webhook]({{< relref path="#create-a-webhook" lang="ko" >}})을 소유한 팀을 선택합니다.
6. **Action type**을 **Webhooks**로 설정한 다음 사용할 [webhook]({{< relref path="#create-a-webhook" lang="ko" >}})을 선택합니다.
7. webhook에 페이로드가 필요한 경우 페이로드를 구성하여 **Payload** 필드에 붙여넣습니다. webhook에 대해 엑세스 토큰을 구성한 경우 `${ACCESS_TOKEN}` [페이로드 변수]({{< relref path="#payload-variables" lang="ko" >}})에서 토큰에 엑세스할 수 있습니다. webhook에 대해 secret을 구성한 경우 이름 앞에 `$`를 붙여 페이로드에서 해당 secret에 엑세스할 수 있습니다. webhook의 요구 사항은 webhook의 서비스에 의해 결정됩니다.
8. **Next step**을 클릭합니다.
9. 자동화 이름을 입력합니다. 선택적으로 설명을 제공합니다. **Create automation**을 클릭합니다.

{{% /tab %}}
{{< /tabpane >}}

## 자동화 보기 및 관리
{{< tabpane text=true >}}
{{% tab "Registry" %}}

- Registry의 **Automations** 탭에서 Registry의 자동화를 관리합니다.
- 컬렉션 세부 정보 페이지의 **Automations** 섹션에서 컬렉션의 자동화를 관리합니다.

이러한 페이지에서 Registry 관리자는 기존 자동화를 관리할 수 있습니다.
- 자동화 세부 정보를 보려면 이름을 클릭합니다.
- 자동화를 편집하려면 해당 액션 `...` 메뉴를 클릭한 다음 **Edit automation**을 클릭합니다.
- 자동화를 삭제하려면 해당 액션 `...` 메뉴를 클릭한 다음 **Delete automation**을 클릭합니다. 확인이 필요합니다.

{{% /tab %}}
{{% tab "Project" %}}
W&B 관리자는 Project의 **Automations** 탭에서 Project의 자동화를 보고 관리할 수 있습니다.

- 자동화 세부 정보를 보려면 이름을 클릭합니다.
- 자동화를 편집하려면 해당 액션 `...` 메뉴를 클릭한 다음 **Edit automation**을 클릭합니다.
- 자동화를 삭제하려면 해당 액션 `...` 메뉴를 클릭한 다음 **Delete automation**을 클릭합니다. 확인이 필요합니다.
{{% /tab %}}
{{< /tabpane >}}

## 페이로드 참조
이 섹션을 사용하여 webhook의 페이로드를 구성합니다. webhook 및 해당 페이로드 테스트에 대한 자세한 내용은 [webhook 문제 해결]({{< relref path="#troubleshoot-your-webhook" lang="ko" >}})을 참조하세요.

### 페이로드 변수
이 섹션에서는 webhook의 페이로드를 구성하는 데 사용할 수 있는 변수에 대해 설명합니다.

| 변수 | 세부 정보 |
|----------|---------|
| `${project_name}` | 액션을 트리거한 변경을 소유한 Project의 이름입니다. |
| `${entity_name}` | 액션을 트리거한 변경을 소유한 엔터티 또는 팀의 이름입니다.
| `${event_type}` | 액션을 트리거한 이벤트 유형입니다. |
| `${event_author}` | 액션을 트리거한 사용자입니다. |
| `${artifact_collection_name}` | 아티팩트 버전이 연결된 아티팩트 컬렉션의 이름입니다. |
| `${artifact_metadata.<KEY>}` | 액션을 트리거한 아티팩트 버전의 임의의 최상위 메타데이터 키의 값입니다. `<KEY>`를 최상위 메타데이터 키의 이름으로 바꿉니다. 최상위 메타데이터 키만 webhook의 페이로드에서 사용할 수 있습니다. |
| `${artifact_version}` | 액션을 트리거한 아티팩트 버전의 [`Wandb.Artifact`]({{< relref path="/ref/python/artifact/" lang="ko" >}}) 표현입니다. |
| `${artifact_version_string}` | 액션을 트리거한 아티팩트 버전의 `string` 표현입니다. |
| `${ACCESS_TOKEN}` | 엑세스 토큰이 구성된 경우 [webhook]({{< relref path="#create-a-webhook" lang="ko" >}})에 구성된 엑세스 토큰의 값입니다. 엑세스 토큰은 `Authorization: Bearer` HTTP 헤더에 자동으로 전달됩니다. |
| `${SECRET_NAME}` | 구성된 경우 [webhook]({{< relref path="#create-a-webhook" lang="ko" >}})에 구성된 secret의 값입니다. `SECRET_NAME`을 secret 이름으로 바꿉니다. |

### 페이로드 예시
이 섹션에는 몇 가지 일반적인 유스 케이스에 대한 webhook 페이로드 예시가 포함되어 있습니다. 이 예시는 [페이로드 변수]({{< relref path="#payload-variables" lang="ko" >}})를 사용하는 방법을 보여줍니다.

{{< tabpane text=true >}}
{{% tab header="GitHub repository dispatch" value="github" %}}

{{% alert %}}
엑세스 토큰에 GHA 워크플로우를 트리거하는 데 필요한 권한 집합이 있는지 확인하세요. 자세한 내용은 [이 GitHub 문서](https://docs.github.com/en/rest/repos/repos?#create-a-repository-dispatch-event)를 참조하세요.
{{% /alert %}}

W&B에서 리포지토리 디스패치를 보내 GitHub 액션을 트리거합니다. 예를 들어 `on` 키에 대한 트리거로 리포지토리 디스패치를 허용하는 GitHub 워크플로우 파일이 있다고 가정합니다.

```yaml
on:
repository_dispatch:
  types: BUILD_AND_DEPLOY
```

리포지토리의 페이로드는 다음과 같을 수 있습니다.

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

렌더링된 템플릿 문자열의 내용과 위치는 자동화가 구성된 이벤트 또는 모델 버전에 따라 달라집니다. `${event_type}`은 `LINK_ARTIFACT` 또는 `ADD_ARTIFACT_ALIAS`로 렌더링됩니다. 아래에서 예시 매핑을 참조하세요.

```text
${event_type} --> "LINK_ARTIFACT" 또는 "ADD_ARTIFACT_ALIAS"
${event_author} --> "<wandb-user>"
${artifact_version} --> "wandb-artifact://_id/QXJ0aWZhY3Q6NTE3ODg5ODg3""
${artifact_version_string} --> "<entity>/model-registry/<registered_model_name>:<alias>"
${artifact_collection_name} --> "<registered_model_name>"
${project_name} --> "model-registry"
${entity_name} --> "<entity>"
```

템플릿 문자열을 사용하여 W&B에서 GitHub Actions 및 기타 툴로 컨텍스트를 동적으로 전달합니다. 이러한 툴이 Python 스크립트를 호출할 수 있는 경우 [W&B API]({{< relref path="/guides/core/artifacts/download-and-use-an-artifact.md" lang="ko" >}})를 통해 등록된 모델 아티팩트를 사용할 수 있습니다.

- 리포지토리 디스패치에 대한 자세한 내용은 [GitHub Marketplace의 공식 문서](https://github.com/marketplace/actions/repository-dispatch)를 참조하세요.

- 모델 평가 및 배포를 위한 자동화를 만드는 방법을 안내하는 동영상 [모델 평가를 위한 Webhook 자동화](https://www.youtube.com/watch?v=7j-Mtbo-E74&ab_channel=Weights%26Biases) 및 [모델 배포를 위한 Webhook 자동화](https://www.youtube.com/watch?v=g5UiAFjM2nA&ab_channel=Weights%26Biases)를 시청하세요.

- Github Actions webhook 자동화를 모델 CI에 사용하는 방법을 보여주는 W&B [리포트](https://wandb.ai/wandb/wandb-model-cicd/reports/Model-CI-CD-with-W-B--Vmlldzo0OTcwNDQw)를 검토하세요. Modal Labs webhook으로 모델 CI를 만드는 방법을 알아보려면 [GitHub 리포지토리](https://github.com/hamelsmu/wandb-modal-webhook)를 확인하세요.

{{% /tab %}}

{{% tab header="Microsoft Teams 알림" value="microsoft"%}}

이 예시 페이로드는 webhook을 사용하여 Teams 채널에 알리는 방법을 보여줍니다.

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

템플릿 문자열을 사용하여 실행 시 W&B 데이터를 페이로드에 삽입할 수 있습니다(위의 Teams 예시 참조).

{{% /tab %}}

{{% tab header="Slack 알림" value="slack"%}}

{{% alert %}}
이 섹션은 과거 기록을 위해 제공됩니다. 현재 webhook을 사용하여 Slack과 통합하는 경우 [새 Slack 인테그레이션]({{ relref "#create-a-slack-automation"}})을 사용하도록 구성을 업데이트하는 것이 좋습니다.
{{% /alert %}}

[Slack API 문서](https://api.slack.com/messaging/webhooks)에 강조 표시된 지침에 따라 Slack 앱을 설정하고 수신 webhook 통합을 추가합니다. `Bot User OAuth Token` 아래에 지정된 secret이 W&B webhook의 엑세스 토큰인지 확인합니다.

다음은 예시 페이로드입니다.

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

## webhook 문제 해결
W&B App UI를 사용하여 대화식으로 또는 Bash 스크립트를 사용하여 프로그래밍 방식으로 webhook 문제를 해결합니다. 새 webhook을 만들거나 기존 webhook을 편집할 때 webhook 문제를 해결할 수 있습니다.

{{< tabpane text=true >}}
{{% tab header="W&B App UI" value="app" %}}

팀 관리자는 W&B App UI를 사용하여 대화식으로 webhook을 테스트할 수 있습니다.

1. W&B 팀 설정 페이지로 이동합니다.
2. **Webhooks** 섹션으로 스크롤합니다.
3. webhook 이름 옆에 있는 가로 세 개의 점(미트볼 아이콘)을 클릭합니다.
4. **Test**를 선택합니다.
5. 나타나는 UI 패널에서 나타나는 필드에 POST 요청을 붙여넣습니다.
    {{< img src="/images/models/webhook_ui.png" alt="Demo of testing a webhook payload" >}}
6. **Test webhook**을 클릭합니다. W&B App UI 내에서 W&B는 엔드포인트에서 응답을 게시합니다.
    {{< img src="/images/models/webhook_ui_testing.gif" alt="Demo of testing a webhook" >}}

시연은 동영상 [Weights & Biases에서 Webhook 테스트](https://www.youtube.com/watch?v=bl44fDpMGJw&ab_channel=Weights%26Biases)를 시청하세요.
{{% /tab %}}

{{% tab header="Bash 스크립트" value="bash"%}}

이 셸 스크립트는 W&B가 트리거될 때 webhook 자동화로 보내는 요청과 유사한 `POST` 요청을 생성하는 한 가지 방법을 보여줍니다.

아래 코드를 복사하여 셸 스크립트에 붙여넣어 webhook 문제를 해결합니다. 다음 값에 대해 사용자 고유의 값을 지정합니다.

* `ACCESS_TOKEN`
* `SECRET`
* `PAYLOAD`
* `API_ENDPOINT`

{{< prism file="/webhook_test.sh" title="webhook_test.sh">}}{{< /prism >}}

{{% /tab %}}
{{< /tabpane >}}

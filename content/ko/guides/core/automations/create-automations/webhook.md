---
title: 웹훅 automation 생성하기
menu:
  default:
    identifier: ko-guides-core-automations-create-automations-webhook
    parent: automations
weight: 3
---

{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-cloud-only.md" >}}
{{% /pageinfo %}}

이 페이지에서는 webhook [automation]({{< relref path="/guides/core/automations/" lang="ko" >}})을 생성하는 방법을 안내합니다. Slack automation을 생성하려면 [Slack automation 생성하기]({{< relref path="/guides/core/automations/create-automations/slack.md" lang="ko" >}}) 문서를 참고하세요.

웹훅 automation을 생성하는 기본적인 단계는 아래와 같습니다.
1. 필요하다면 automation에서 사용하는 민감한 문자열(예: 엑세스 토큰, 비밀번호, 또는 SSH 키)마다 [W&B secret 생성]({{< relref path="/guides/core/secrets.md" lang="ko" >}})을 합니다. Secret은 **Team Settings**에서 정의됩니다.
1. [Webhooks 생성]({{< relref path="#create-a-webhook" lang="ko" >}})을 통해 endpoint와 인증 정보를 정의하고, 인테그레이션이 필요한 secret에 엑세스할 수 있도록 허용합니다.
1. [Automation 생성]({{< relref path="#create-an-automation" lang="ko" >}})에서 감지할 [event]({{< relref path="/guides/core/automations/automation-events.md" lang="ko" >}})와 W&B가 전송할 payload를 지정합니다. payload에 필요한 secret 엑세스 권한도 부여해야 합니다.

## Webhook 생성하기
Team admin은 팀 단위로 webhook을 추가할 수 있습니다.

{{% alert %}}
Webhook이 Bearer 토큰이나 payload에 민감한 정보가 필요하다면, webhook을 생성하기 전에 [secret을 먼저 등록]({{< relref path="/guides/core/secrets.md#add-a-secret" lang="ko" >}})하세요. Webhook 당 엑세스 토큰과 다른 secret은 각각 최대 하나씩만 지정할 수 있습니다. 인증 및 권한 부여 요구사항은 webhook 서비스에 따라 결정됩니다.
{{% /alert %}}

1. W&B에 로그인 후 **Team Settings** 페이지로 이동합니다.
1. **Webhooks** 섹션에서 **New webhook**을 클릭합니다.
1. webhook에 사용할 이름을 지정합니다.
1. webhook의 endpoint URL을 입력합니다.
1. Bearer 토큰이 필요한 경우, **Access token**에 해당하는 [secret]({{< relref path="/guides/core/secrets.md" lang="ko" >}})을 지정하세요. webhook automation 사용 시, W&B는 `Authorization: Bearer` HTTP 헤더에 엑세스 토큰을 설정하고, `${ACCESS_TOKEN}` [payload 변수]({{< relref path="#payload-variables" lang="ko" >}})로 토큰을 엑세스할 수 있습니다. W&B가 webhook 서비스로 보내는 `POST` 요청 구조는 [Webhook 문제 해결]({{< relref path="#troubleshoot-your-webhook" lang="ko" >}})에서 자세히 확인할 수 있습니다.
1. webhook payload에 비밀번호 등 민감한 정보가 요구된다면 **Secret**에 해당 secret을 지정하세요. webhook을 사용하는 automation을 설정할 때, secret 이름 앞에 `$`를 붙여 [payload 변수]({{< relref path="#payload-variables" lang="ko" >}})로 사용할 수 있습니다.

    webhook의 access token이 secret에 저장되어 있다면, access token으로 secret을 지정하는 다음 단계도 반드시 진행해야 합니다.
1. W&B가 endpoint에 정상적으로 연결되고 인증할 수 있는지 확인하려면:
    1. 필요하다면, 테스트용 payload를 입력하세요. payload 내에서 webhook이 엑세스할 수 있는 secret을 사용하려면 name 앞에 `$`를 붙이세요. 이 payload는 테스트에만 사용되며 저장되지 않습니다. automation의 payload는 [automation 생성 시]({{< relref path="#create-a-webhook-automation" lang="ko" >}}) 따로 지정합니다. secret과 엑세스 토큰이 `POST` 요청의 어디에 위치하는지는 [Webhook 문제 해결]({{< relref path="#troubleshoot-your-webhook" lang="ko" >}}) 문서를 참고하세요.
    1. **Test**를 클릭하면, W&B가 입력한 자격증명을 사용해 webhook endpoint에 연결을 시도합니다. payload가 있다면 함께 전송합니다.

    테스트에 실패한다면 webhook 설정을 다시 한 번 점검해 보시고, 필요하다면 [Webhook 문제 해결]({{< relref path="#troubleshoot-your-webhook" lang="ko" >}}) 문서를 참고하세요.

![Screenshot showing two webhooks in a Team](/images/automations/webhooks.png)

이제 webhook을 사용하는 [automation을 생성]({{< relref path="#create-a-webhook-automation" lang="ko" >}})할 수 있습니다.

## Automation 생성하기
[Webhook을 설정]({{< relref path="#create-a-webhook" lang="ko" >}}) 한 후, **Registry** 또는 **Project**를 선택해서 webhook을 트리거하는 automation을 만듭니다.

{{< tabpane text=true >}}
{{% tab "Registry" %}}
Registry admin은 해당 registry에서 automation을 생성할 수 있습니다. Registry automation은 앞으로 추가되는 컬렉션을 포함해 registry 내 모든 컬렉션에 적용됩니다.

1. W&B에 로그인합니다.
1. registry 이름을 클릭해 상세 페이지로 이동합니다.
1. registry 단위로 automation을 생성하려면 **Automations** 탭에서 **Create automation**을 클릭합니다. Registry에 적용된 automation은 해당 registry의 모든 컬렉션(이후 생성된 컬렉션 포함)에 자동으로 적용됩니다.

    registry 내 특정 컬렉션에만 automation을 적용하려면 컬렉션의 `...` 메뉴에서 **Create automation**을 클릭하세요. 또는 컬렉션 상세 페이지의 **Automations** 섹션에서 **Create automation**을 클릭해도 됩니다.
1. 감지할 [event]({{< relref path="/guides/core/automations/automation-events.md" lang="ko" >}})를 선택하고, event에 따라 추가로 표시되는 필드를 입력하세요. 예를 들어 **An artifact alias is added**를 선택하면 **Alias regex**를 지정해야 합니다. **Next step**을 클릭하세요.
1. webhook을 소유한 team을 선택하세요.
1. **Action type**을 **Webhooks**로 설정한 후 사용할 [webhook]({{< relref path="#create-a-webhook" lang="ko" >}})을 선택하세요.
1. webhook에 엑세스 토큰이 지정되어 있다면, `${ACCESS_TOKEN}` [payload 변수]({{< relref path="#payload-variables" lang="ko" >}})로 토큰을 사용할 수 있습니다. webhook의 secret도 payload에 name 앞에 `$`를 붙여 사용 가능합니다. webhook의 요구사항은 해당 서비스에 의해 결정됩니다.
1. **Next step**을 클릭하세요.
1. automation의 이름을 입력하고, 필요하다면 설명도 작성하세요. **Create automation**을 클릭합니다.

{{% /tab %}}
{{% tab "Project" %}}
W&B admin은 프로젝트에서 automation을 만들 수 있습니다.

1. W&B에 로그인 후 프로젝트 페이지로 이동하세요.
1. 사이드바에서 **Automations**를 클릭한 다음 **Create automation**을 클릭합니다.

    또는 워크스페이스 내 라인 플롯에서, 패널 상단의 알림(bell) 아이콘을 클릭해 해당 metric에 대한 [run metric automation]({{< relref path="/guides/core/automations/automation-events.md#run-events" lang="ko" >}})을 빠르게 생성할 수 있습니다.
    {{< img src="/images/automations/run_metric_automation_from_panel.png" alt="Automation bell icon location" >}}
1. 감지할 [event]({{< relref path="/guides/core/automations/automation-events.md" lang="ko" >}})를 선택하세요(예: artifact alias가 추가될 때, run metric이 일정 기준을 충족할 때 등).

    1. 선택한 event에 따라 추가로 표시되는 필드를 입력하세요. 예시로 **An artifact alias is added**를 선택하면 **Alias regex**를 지정해야 합니다.

    1. 필요하다면 collection filter도 입력하세요. 그렇지 않으면 automation은 프로젝트 내 모든 컬렉션에 적용됩니다(추후 추가 컬렉션 포함).

    **Next step**을 클릭하세요.
1. webhook을 소유한 team을 선택하세요.
1. **Action type**을 **Webhooks**로 설정 후 사용할 [webhook]({{< relref path="#create-a-webhook" lang="ko" >}})을 선택하세요.
1. webhook에서 payload가 필요하다면 직접 payload를 작성해 **Payload** 필드에 붙여넣으세요. 엑세스 토큰이 있으면 `${ACCESS_TOKEN}` [payload 변수]({{< relref path="#payload-variables" lang="ko" >}})로, secret이 있으면 이름 앞에 `$`를 붙여 payload에 포함할 수 있습니다. webhook의 요구사항은 해당 서비스에 따라 달라집니다.
1. **Next step**을 클릭하세요.
1. automation의 이름을 입력하시고, 필요하면 설명을 추가하세요. **Create automation**을 클릭하세요.

{{% /tab %}}
{{< /tabpane >}}

## Automation 확인 및 관리
{{< tabpane text=true >}}
{{% tab "Registry" %}}

- registry의 **Automations** 탭에서 registry 전체 automation을 관리할 수 있습니다.
- 컬렉션 상세 페이지의 **Automations** 섹션에서 해당 컬렉션의 automation을 관리할 수 있습니다.

Registry admin은 위 페이지에서 기존 automation을 관리할 수 있습니다.
- automation 상세정보를 보려면 이름을 클릭하세요.
- automation 수정을 위해서는 `...` 메뉴에서 **Edit automation**을 클릭하세요.
- automation을 삭제하려면 `...` 메뉴에서 **Delete automation**을 클릭하세요. 삭제시 확인이 필요합니다.

{{% /tab %}}
{{% tab "Project" %}}
W&B admin은 프로젝트 **Automations** 탭에서 프로젝트의 automation을 확인하고 관리할 수 있습니다.

- automation 상세정보를 보려면 이름을 클릭하세요.
- automation 수정을 위해서는 액션의 `...` 메뉴에서 **Edit automation**을 클릭하세요.
- automation을 삭제하려면 액션의 `...` 메뉴에서 **Delete automation**을 클릭하세요. 삭제 시에는 확인이 필요합니다.
{{% /tab %}}
{{< /tabpane >}}

## Payload 참고
이 섹션은 webhook payload를 구성할 때 참고할 수 있습니다. webhook 및 payload 테스트 방법은 [Webhook 문제 해결]({{< relref path="#troubleshoot-your-webhook" lang="ko" >}})에서 확인할 수 있습니다.

### Payload 변수
webhook의 payload를 만들 때 사용할 수 있는 변수들입니다.

| 변수 | 설명 |
|----------|---------|
| `${project_name}`             | 액션을 트리거한 mutation을 소유한 프로젝트 이름 |
| `${entity_name}`              | 액션을 트리거한 mutation을 소유한 Entity 혹은 Team 이름 |
| `${event_type}`               | 액션을 트리거한 이벤트 종류 |
| `${event_author}`             | 액션을 트리거한 유저 |
| `${alias}`                    | **An artifact alias is added** 이벤트로 automation이 동작한 경우 artifact의 alias 값. 나머지 automation에는 빈 문자열입니다. |
| `${tag}`                      | **An artifact tag is added** 이벤트로 automation이 동작한 경우 artifact의 tag 값. 나머지 automation에는 빈 문자열입니다. |
| `${artifact_collection_name}` | artifact 버전이 연결된 artifact 컬렉션 이름 |
| `${artifact_metadata.<KEY>}`  | 액션을 트리거한 artifact 버전의 최상위 메타데이터 키의 값. `<KEY>`를 실제 메타데이터 키 이름으로 대체하세요. webhook payload에서는 최상위 메타데이터만 제공됩니다. |
| `${artifact_version}`         | 해당 액션을 트리거한 artifact 버전의 [`Wandb.Artifact`]({{< relref path="/ref/python/sdk/classes/artifact.md/" lang="ko" >}}) 객체 표현 |
| `${artifact_version_string}` | 해당 액션을 트리거한 artifact 버전의 `string` 표현 |
| `${ACCESS_TOKEN}` | [webhook]({{< relref path="#create-a-webhook" lang="ko" >}})에 엑세스 토큰이 지정된 경우 그 값. 엑세스 토큰은 `Authorization: Bearer` HTTP 헤더로 자동 전달됩니다. |
| `${SECRET_NAME}` | webhook에 지정된 secret이 있다면, 값이 여기에 들어갑니다. `SECRET_NAME`을 실제 secret 명칭으로 바꿔 사용하세요. |

### Payload 예시
일반적인 유스 케이스별로 webhook payload 예시를 제공합니다. 각각의 예시는 [payload 변수]({{< relref path="#payload-variables" lang="ko" >}}) 사용 방법을 보여줍니다.

{{< tabpane text=true >}}
{{% tab header="GitHub repository dispatch" value="github" %}}

{{% alert %}}
엑세스 토큰에 GHA 워크플로우를 트리거할 수 있는 권한이 있는지 꼭 확인하세요. 자세한 내용은 [GitHub Docs 참고](https://docs.github.com/en/rest/repos/repos?#create-a-repository-dispatch-event).
{{% /alert %}}

W&B에서 repository dispatch를 송신해 GitHub Action을 트리거할 수 있습니다. 예시로, 아래와 같이 `on` 키에 repository dispatch trigger가 있는 workflow 파일이 있다고 가정해 봅니다.

```yaml
on:
repository_dispatch:
  types: BUILD_AND_DEPLOY
```

repository에 전송할 payload 예시는 다음과 같습니다.

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
webhook payload의 `event_type` 키 값은 GitHub workflow YAML 파일 내의 `types` 필드와 일치해야 합니다.
{{% /alert %}}

템플릿 문자열이 어떻게 치환되는지, 어떤 이벤트 또는 모델 버전 automation에서 사용되는지에 따라 다릅니다. `${event_type}`은 `LINK_ARTIFACT` 또는 `ADD_ARTIFACT_ALIAS`로 렌더링될 수 있습니다. 아래와 같이 매핑 예시를 참고하세요.

```text
${event_type} --> "LINK_ARTIFACT" 또는 "ADD_ARTIFACT_ALIAS"
${event_author} --> "<wandb-user>"
${artifact_version} --> "wandb-artifact://_id/QXJ0aWZhY3Q6NTE3ODg5ODg3""
${artifact_version_string} --> "<entity>/model-registry/<registered_model_name>:<alias>"
${artifact_collection_name} --> "<registered_model_name>"
${project_name} --> "model-registry"
${entity_name} --> "<entity>"
```

템플릿 문자열을 활용해 W&B의 context 정보를 동적으로 GitHub Actions 등 외부 툴로 전달할 수 있습니다. 외부 툴이 Python 스크립트를 실행할 수 있다면, 해당 registered model artifacts를 [W&B API]({{< relref path="/guides/core/artifacts/download-and-use-an-artifact.md" lang="ko" >}})로 활용할 수 있습니다.

- repository dispatch에 대한 자세한 정보는 [GitHub Marketplace 공식 문서](https://github.com/marketplace/actions/repository-dispatch)를 참고하세요.

- 모델 평가와 배포 automation에 대한 가이드는 다음 동영상을 참고하세요: [Webhook Automations for Model Evaluation](https://www.youtube.com/watch?v=7j-Mtbo-E74&ab_channel=Weights%26Biases), [Webhook Automations for Model Deployment](https://www.youtube.com/watch?v=g5UiAFjM2nA&ab_channel=Weights%26Biases)

- Model CI automation에 활용된 예시를 [W&B report](https://wandb.ai/wandb/wandb-model-cicd/reports/Model-CI-CD-with-W-B--Vmlldzo0OTcwNDQw)에서 확인할 수 있습니다. [GitHub repository](https://github.com/hamelsmu/wandb-modal-webhook)를 참고해 Modal Labs webhook으로 Model CI를 구현하는 방법도 살펴보세요.

{{% /tab %}}

{{% tab header="Microsoft Teams notification" value="microsoft"%}}

아래는 Teams channel로 webhook을 통해 알림을 전송하는 예시 payload입니다.

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

위 예시처럼 템플릿 문자열을 사용해 W&B 데이터를 payload에 실행 시점에 넣을 수 있습니다.

{{% /tab %}}

{{% tab header="Slack notifications" value="slack"%}}

{{% alert %}}
이 섹션은 기존 webhook-Slack 인테그레이션 사용자의 편의를 위해 안내됩니다. webhook으로 Slack과 연동 중이라면, [새로운 Slack 인테그레이션]({{ relref "#create-a-slack-automation"}}) 사용으로 업데이트하는 것을 권장합니다.
{{% /alert %}}

Slack 앱을 설정하고, [Slack API 문서](https://api.slack.com/messaging/webhooks)에 안내된 대로 incoming webhook 인테그레이션을 추가하세요. secret으로 설정된 `Bot User OAuth Token`을 W&B webhook의 엑세스 토큰으로 지정해야 합니다.

payload 예시는 아래와 같습니다.

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

## Webhook 문제 해결하기
W&B App UI를 사용하거나 bash 스크립트로 webhook을 인터랙티브하게 점검할 수 있습니다. webhook은 신규 생성 시 또는 기존 webhook 수정 시 테스트할 수 있습니다.

W&B가 `POST` 요청에 사용하는 포맷이 궁금하다면 **Bash script** 탭을 참고하세요.

{{< tabpane text=true >}}
{{% tab header="W&B App UI" value="app" %}}

team admin은 W&B App UI로 webhook을 직접 테스트할 수 있습니다.

1. W&B의 Team Settings 페이지로 이동하세요.
2. **Webhooks** 섹션까지 스크롤합니다.
3. webhook 이름 옆의 가로 점 세 개(미트볼 아이콘)을 클릭합니다.
4. **Test**를 선택합니다.
5. 표시되는 패널 UI에서 요청할 POST content를 붙여넣습니다.
    {{< img src="/images/models/webhook_ui.png" alt="Demo of testing a webhook payload" >}}
6. **Test webhook**을 클릭하면, W&B App UI에서 endpoint의 응답이 바로 표시됩니다.
    {{< img src="/images/models/webhook_ui_testing.gif" alt="Demo of testing a webhook" >}}

실습 영상 [Testing Webhooks in W&B](https://www.youtube.com/watch?v=bl44fDpMGJw&ab_channel=Weights%26Biases)도 참고할 수 있습니다.
{{% /tab %}}

{{% tab header="Bash script" value="bash"%}}

아래 쉘 스크립트 예시는 W&B가 webhook automation 트리거 시 보내는 `POST` 요청과 유사한 형식을 직접 만들어 호출하는 방법을 보여줍니다.

아래 코드를 복사해 쉘 스크립트로 붙여넣고, 다음 값을 직접 지정하세요:

* `ACCESS_TOKEN`
* `SECRET`
* `PAYLOAD`
* `API_ENDPOINT`

{{< prism file="/webhook_test.sh" title="webhook_test.sh">}}{{< /prism >}}

{{% /tab %}}
{{< /tabpane >}}
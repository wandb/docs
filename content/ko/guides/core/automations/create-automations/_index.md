---
title: 자동화 생성
menu:
  default:
    identifier: ko-guides-core-automations-create-automations-_index
    parent: automations
weight: 1
---

{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-cloud-only.md" >}}
{{% /pageinfo %}}

이 페이지에서는 W&B [automations]({{< relref path="/guides/core/automations/" lang="ko" >}}) 생성 및 관리 개요를 제공합니다. 자세한 사용 방법은 [Slack automation 생성하기]({{< relref path="/guides/core/automations/create-automations/slack.md" lang="ko" >}}) 또는 [webhook automation 생성하기]({{< relref path="/guides/core/automations/create-automations/webhook.md" lang="ko" >}})를 참고하세요.

{{% alert %}}
automations 관련 튜토리얼을 찾고 계신가요?
- [모델 평가 및 배포를 위한 Github Action을 자동으로 트리거하는 방법 알아보기](https://wandb.ai/wandb/wandb-model-cicd/reports/Model-CI-CD-with-W-B--Vmlldzo0OTcwNDQw).
- [모델을 Sagemaker endpoint에 자동 배포하는 데모 영상 보기](https://www.youtube.com/watch?v=s5CMj_w3DaQ).
- [automations 소개 영상 시리즈 시청하기](https://youtube.com/playlist?list=PLD80i8An1OEGECFPgY-HPCNjXgGu-qGO6&feature=shared).
{{% /alert %}}

## 요구 사항
- 팀 관리자는 해당 팀의 Projects와 automation에 필요한 구성 요소(예: webhooks, secrets, Slack integration)를 생성하고 관리할 수 있습니다. [Team settings]({{< relref path="/guides/models/app/settings-page/team-settings/" lang="ko" >}}) 참고.
- registry automation을 생성하려면 registry에 대한 엑세스 권한이 필요합니다. [Registry 엑세스 설정]({{< relref path="/guides/core/registry/configure_registry.md#registry-roles" lang="ko" >}})을 참고하세요.
- Slack automation을 만들려면 선택한 Slack 인스턴스와 채널에 게시할 수 있는 권한이 필요합니다.

## automation 생성하기
Project 또는 registry의 **Automations** 탭에서 automation을 생성할 수 있습니다. 전반적인 절차는 다음과 같습니다:

1. 필요한 경우, automation에서 사용하는 민감 정보(엑세스 토큰, 비밀번호, SSH 키 등)마다 [W&B secret 생성]({{< relref path="/guides/core/secrets.md" lang="ko" >}})을 만듭니다. Secret은 **Team Settings**에서 정의합니다. webhook automation에서 가장 자주 사용됩니다.
1. webhook 또는 Slack integration을 설정하여 W&B가 Slack에 게시하거나 webhook을 대신 실행할 수 있도록 권한을 부여합니다. 하나의 webhook 또는 Slack integration은 여러 automation에서 재사용할 수 있습니다. 이 작업은 **Team Settings**에서 진행합니다.
1. Project 또는 registry에서 automation을 생성하고, 감시할 이벤트와 실행할 작업(예: Slack에 메시지 보내기, webhook 실행 등)을 지정합니다. webhook automation을 만들 경우 전송할 payload도 설정합니다.

또한, workspace 내 line plot에서 해당 지표에 대해 [run metric automation]({{< relref path="/guides/core/automations/automation-events.md#run-events" lang="ko" >}})을 빠르게 생성할 수 있습니다:

1. 패널에 마우스를 올린 후, 패널 상단의 종(bell) 아이콘을 클릭하세요.

    {{< img src="/images/automations/run_metric_automation_from_panel.png" alt="Automation bell icon location" >}}
1. 기본 또는 고급 설정을 활용하여 automation을 구성합니다. 예를 들어 run filter를 적용하여 실행 대상 범위를 좁히거나, 절대 임계값을 설정할 수 있습니다.

자세한 내용은 아래를 참고하세요:

- [Slack automation 생성하기]({{< relref path="slack.md" lang="ko" >}})
- [webhook automation 생성하기]({{< relref path="webhook.md" lang="ko" >}})

## automations 조회 및 관리
Project 또는 registry의 **Automations** 탭에서 automations를 조회·관리할 수 있습니다.

- automation의 상세 정보를 보려면 이름을 클릭하세요.
- automation을 수정하려면 해당 항목의 `...` 메뉴에서 **Edit automation**을 선택하세요.
- automation을 삭제하려면 해당 항목의 `...` 메뉴에서 **Delete automation**을 클릭하세요.

## 다음 단계
- [automation 이벤트 및 범위에 대해 더 알아보기]({{< relref path="/guides/core/automations/automation-events.md" lang="ko" >}})
- [Slack automation 생성하기]({{< relref path="slack.md" lang="ko" >}}).
- [webhook automation 생성하기]({{< relref path="webhook.md" lang="ko" >}}).
- [secret 생성하기]({{< relref path="/guides/core/secrets.md" lang="ko" >}}).
---
title: Create an automation
menu:
  default:
    identifier: ko-guides-core-automations-create-automations-_index
    parent: automations
weight: 1
---

{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-cloud-only.md" >}}
{{% /pageinfo %}}

이 페이지에서는 W&B [자동화]({{< relref path="/guides/core/automations/" lang="ko" >}}) 생성 및 관리에 대한 개요를 제공합니다. 자세한 내용은 [Slack 자동화 생성]({{< relref path="/guides/core/automations/create-automations/slack.md" lang="ko" >}}) 또는 [Webhook 자동화 생성]({{< relref path="/guides/core/automations/create-automations/webhook.md" lang="ko" >}})을 참조하세요.

{{% alert %}}
자동화 관련 튜토리얼을 찾고 계신가요?
- [모델 평가 및 배포를 위해 Github Action을 자동으로 트리거하는 방법 알아보기](https://wandb.ai/wandb/wandb-model-cicd/reports/Model-CI-CD-with-W-B--Vmlldzo0OTcwNDQw).
- [모델을 Sagemaker 엔드포인트에 자동으로 배포하는 방법을 보여주는 비디오 시청하기](https://www.youtube.com/watch?v=s5CMj_w3DaQ).
- [자동화를 소개하는 비디오 시리즈 시청하기](https://youtube.com/playlist?list=PLD80i8An1OEGECFPgY-HPCNjXgGu-qGO6&feature=shared).
{{% /alert %}}

## 요구 사항
- 팀 관리자는 팀의 Projects에 대한 자동화와 웹훅, 보안 비밀 또는 Slack 연결과 같은 자동화 구성 요소를 생성하고 관리할 수 있습니다. [팀 설정]({{< relref path="/guides/models/app/settings-page/team-settings/" lang="ko" >}})을 참조하세요.
- 레지스트리 자동화를 생성하려면 레지스트리에 대한 엑세스 권한이 있어야 합니다. [레지스트리 엑세스 구성]({{< relref path="/guides/core/registry/configure_registry.md#registry-roles" lang="ko" >}})을 참조하세요.
- Slack 자동화를 생성하려면 선택한 Slack 인스턴스 및 채널에 게시할 수 있는 권한이 있어야 합니다.

## 자동화 생성
Project 또는 레지스트리의 **Automations** 탭에서 자동화를 생성합니다. 개략적으로 자동화를 생성하려면 다음 단계를 따르세요.

1. 필요한 경우 엑세스 토큰, 비밀번호 또는 SSH 키와 같이 자동화에 필요한 각 민감한 문자열에 대해 [W&B 보안 비밀 생성]({{< relref path="/guides/core/secrets.md" lang="ko" >}})을 수행합니다. 보안 비밀은 **Team Settings**에 정의되어 있습니다. 보안 비밀은 일반적으로 웹훅 자동화에서 사용됩니다.
2. W&B가 Slack에 게시하거나 사용자를 대신하여 웹훅을 실행할 수 있도록 웹훅 또는 Slack 알림을 구성합니다. 단일 자동화 작업(웹훅 또는 Slack 알림)은 여러 자동화에서 사용할 수 있습니다. 이러한 작업은 **Team Settings**에 정의되어 있습니다.
3. Project 또는 레지스트리에서 감시할 이벤트와 수행할 작업(예: Slack에 게시 또는 웹훅 실행)을 지정하는 자동화를 생성합니다. 웹훅 자동화를 생성할 때 전송할 페이로드를 구성합니다.

자세한 내용은 다음을 참조하세요.

- [Slack 자동화 생성]({{< relref path="slack.md" lang="ko" >}})
- [Webhook 자동화 생성]({{< relref path="webhook.md" lang="ko" >}})

## 자동화 보기 및 관리
Project 또는 레지스트리의 **Automations** 탭에서 자동화를 보고 관리합니다.

- 자동화 세부 정보를 보려면 해당 이름을 클릭합니다.
- 자동화를 편집하려면 해당 작업 `...` 메뉴를 클릭한 다음 **자동화 편집**을 클릭합니다.
- 자동화를 삭제하려면 해당 작업 `...` 메뉴를 클릭한 다음 **자동화 삭제**를 클릭합니다.

## 다음 단계
- [Slack 자동화 생성]({{< relref path="slack.md" lang="ko" >}}).
- [Webhook 자동화 생성]({{< relref path="webhook.md" lang="ko" >}}).
- [보안 비밀 생성]({{< relref path="/guides/core/secrets.md" lang="ko" >}}).

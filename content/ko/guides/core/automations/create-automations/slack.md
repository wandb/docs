---
title: Slack automation 만들기
menu:
  default:
    identifier: ko-guides-core-automations-create-automations-slack
    parent: create-automations
weight: 1
---

{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-cloud-only.md" >}}
{{% /pageinfo %}}

이 페이지에서는 Slack [automation]({{< relref path="/guides/core/automations/" lang="ko" >}}> )을 만드는 방법을 안내합니다. 웹훅 automation을 만들고 싶다면 [웹훅 automation 생성]({{< relref path="/guides/core/automations/create-automations/webhook.md" lang="ko" >}}) 가이드를 참고하세요.

간략하게, Slack automation을 만들려면 다음 단계로 진행합니다:
1. [Slack 인테그레이션 추가]({{< relref path="#add-a-slack-integration" lang="ko" >}}): W&B 에서 해당 Slack 인스턴스와 채널에 게시할 수 있도록 권한을 부여합니다.
1. [automation 생성]({{< relref path="#create-an-automation" lang="ko" >}}): 감지할 [이벤트]({{< relref path="/guides/core/automations/automation-events.md" lang="ko" >}}) 와 알릴 채널을 정의합니다.

## Slack 인테그레이션 추가
팀 관리자는 팀에 Slack 인테그레이션을 추가할 수 있습니다.

1. W&B 에 로그인하고 **Team Settings**로 이동합니다.
1. **Slack channel integrations** 섹션에서 **Connect Slack**을 클릭해 새로운 Slack 인스턴스를 추가합니다. 이미 연동된 Slack 인스턴스에 채널을 추가하려면 **New integration**을 클릭하세요.

    ![팀의 두 개 Slack 인테그레이션이 표시된 스크린샷](/images/automations/slack_integrations.png)
1. 필요하다면 브라우저에서 Slack에 로그인합니다. 안내에 따라 W&B 가 선택한 Slack 채널에 게시할 수 있도록 허용하세요. 해당 페이지를 읽은 뒤 **Search for a channel**을 클릭하고 채널 이름을 입력하기 시작하세요. 목록에서 채널을 선택한 후 **Allow**를 클릭하세요.
1. Slack에서 선택한 채널로 이동하세요. 만약 `[Your Slack handle]님이 이 채널에 Weights & Biases 인테그레이션을 추가했습니다.`와 같은 게시글이 보이면, 인테그레이션 연결이 정상적으로 완료된 것입니다.

이제 설정한 Slack 채널에 알림을 보내는 [automation]({{< relref path="#create-an-automation" lang="ko" >}})을 만들 수 있습니다.

## Slack 인테그레이션 보기 및 관리
팀 관리자는 팀의 Slack 인스턴스와 채널을 확인 및 관리할 수 있습니다.

1. W&B 에 로그인하고 **Team Settings**로 이동합니다.
1. **Slack channel integrations** 섹션에서 각 Slack 목적지를 확인합니다.
1. 목록에서 목적지 오른쪽에 있는 휴지통 아이콘을 클릭해 삭제합니다.

## automation 생성
[Slack 인테그레이션 추가]({{< relref path="#add-a-slack-integreation" lang="ko" >}}) 후에, **Registry**나 **Project**를 선택하고 아래의 단계를 따라 해당 Slack 채널에 알림을 보내는 automation을 만들어보세요.

{{< tabpane text=true >}}
{{% tab "Registry" %}}
Registry 관리자는 해당 registry 내에서 automation을 만들 수 있습니다.

1. W&B 에 로그인합니다.
1. 상세 정보 보기를 원하는 registry 이름을 클릭합니다.
1. Registry에 적용되는 automation을 만들려면 **Automations** 탭에서 **Create automation**을 클릭하세요. registry 범위의 automation은 해당 registry에 속한 모든 collection(앞으로 생성될 collection 포함)에 자동으로 적용됩니다.

    특정 collection에만 적용할 automation을 만들고 싶다면, collection의 `...` 메뉴에서 **Create automation**을 클릭하세요. 또는, collection 상세 페이지의 **Automations** 섹션에서 **Create automation** 버튼을 통해 automation을 만들 수 있습니다.
1. 감지할 [이벤트]({{< relref path="/guides/core/automations/automation-events.md" lang="ko" >}})를 선택하세요.

    이벤트 설정에 따라 추가 입력 필드가 나타날 수 있습니다. 예를 들어, **An artifact alias is added**를 선택하면 **Alias regex**를 입력해야 합니다.

    **Next step**을 클릭하세요.
1. [Slack 인테그레이션]({{< relref path="#add-a-slack-integration" lang="ko" >}})이 속한 팀을 선택하세요.
1. **Action type**을 **Slack notification**으로 설정하고 Slack 채널을 선택한 뒤 **Next step**을 클릭하세요.
1. automation의 이름을 입력하세요. 필요하다면 설명도 추가할 수 있습니다.
1. **Create automation**을 클릭하세요.

{{% /tab %}}
{{% tab "Project" %}}
W&B 관리자는 프로젝트 내에서 automation을 만들 수 있습니다.

1. W&B 에 로그인하세요.
1. 프로젝트 페이지로 이동해 **Automations** 탭을 클릭하고, **Create automation**을 클릭하세요.

    또는 워크스페이스의 라인 플롯 패널에서 해당 metric에 대한 [run metric automation]({{< relref path="/guides/core/automations/automation-events.md#run-events" lang="ko" >}})을 빠르게 생성할 수 있습니다. 패널에 마우스를 올린 뒤 상단의 종(bell) 아이콘을 클릭하세요.
    {{< img src="/images/automations/run_metric_automation_from_panel.png" alt="Automation bell icon location" >}}
1. 감지할 [이벤트]({{< relref path="/guides/core/automations/automation-events.md" lang="ko" >}})를 선택하세요.

    이벤트 설정에 따라 추가 입력 필드가 나타날 수 있습니다. 예를 들어, **An artifact alias is added**를 선택하면 **Alias regex**를 입력해야 합니다.

    **Next step**을 클릭하세요.
1. [Slack 인테그레이션]({{< relref path="#add-a-slack-integration" lang="ko" >}})이 속한 팀을 선택하세요.
1. **Action type**을 **Slack notification**으로 설정하고 Slack 채널을 선택한 뒤 **Next step**을 클릭하세요.
1. automation의 이름을 입력하세요. 필요하다면 설명도 추가할 수 있습니다.
1. **Create automation**을 클릭하세요.

{{% /tab %}}
{{< /tabpane >}}

## automation 보기 및 관리

{{< tabpane text=true >}}
{{% tab "Registry" %}}

- registry의 **Automations** 탭에서 registry에 속한 automation를 관리할 수 있습니다.
- collection의 경우, 해당 collection 상세 페이지의 **Automations** 섹션에서 automation를 관리하세요.

이 두 페이지 중 하나에서 Registry 관리자는 기존 automation를 다음과 같이 관리할 수 있습니다:
- automation 상세 정보를 보려면 이름을 클릭하세요.
- automation를 수정하려면 해당 `...` 메뉴에서 **Edit automation**을 클릭하세요.
- automation를 삭제하려면 해당 `...` 메뉴에서 **Delete automation**을 클릭하세요. 삭제 시 확인이 필요합니다.

{{% /tab %}}
{{% tab "Project" %}}
W&B 관리자는 프로젝트의 **Automations** 탭에서 프로젝트 automation를 확인 및 관리할 수 있습니다.

- automation 상세 정보를 보려면 이름을 클릭하세요.
- automation를 수정하려면 해당 `...` 메뉴에서 **Edit automation**을 클릭하세요.
- automation를 삭제하려면 해당 `...` 메뉴에서 **Delete automation**을 클릭하세요. 삭제 시 확인이 필요합니다.
{{% /tab %}}
{{< /tabpane >}}
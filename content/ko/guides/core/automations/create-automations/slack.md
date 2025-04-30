---
title: Create a Slack automation
menu:
  default:
    identifier: ko-guides-core-automations-create-automations-slack
    parent: create-automations
weight: 1
---

{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-cloud-only.md" >}}
{{% /pageinfo %}}

이 페이지에서는 Slack [자동화]({{< relref path="/guides/core/automations/" lang="ko" >}})를 만드는 방법을 보여줍니다. 웹훅 자동화를 만들려면 [웹훅 자동화 생성]({{< relref path="/guides/core/automations/create-automations/webhook.md" lang="ko" >}})를 대신 참조하세요.

Slack 자동화를 생성하려면 다음과 같은 단계를 수행합니다.
1. Slack [통합 추가]({{< relref path="#add-a-slack-channel" lang="ko" >}})를 통해 Weights & Biases가 Slack 인스턴스 및 채널에 게시할 수 있도록 승인합니다.
1. [Slack 자동화 생성]({{< relref path="#create-slack-automation" lang="ko" >}})를 통해 감시할 [이벤트]({{< relref path="/guides/core/automations/automation-events.md" lang="ko" >}})와 게시할 채널을 정의합니다.

## Slack에 연결
팀 관리자는 팀에 Slack 대상을 추가할 수 있습니다.

1. W&B에 로그인하고 팀 설정 페이지로 이동합니다.
2. **Slack 채널 통합** 섹션에서 **Slack 연결**을 클릭하여 새 Slack 인스턴스를 추가합니다. 기존 Slack 인스턴스에 대한 채널을 추가하려면 **새로운 통합**을 클릭합니다.

    필요한 경우 브라우저에서 Slack에 로그인합니다. 메시지가 표시되면 선택한 Slack 채널에 게시할 수 있는 권한을 W&B에 부여합니다. 페이지를 읽은 다음 **채널 검색**을 클릭하고 채널 이름을 입력하기 시작합니다. 목록에서 채널을 선택한 다음 **허용**을 클릭합니다.

3. Slack에서 선택한 채널로 이동합니다. `[Your Slack handle] added an integration to this channel: Weights & Biases`와 같은 게시물이 표시되면 통합이 올바르게 구성된 것입니다.

이제 구성한 Slack 채널에 알림을 보내는 [자동화 생성]({{< relref path="#create-a-slack-automation" lang="ko" >}})를 할 수 있습니다.

## Slack 연결 보기 및 관리
팀 관리자는 팀의 Slack 인스턴스 및 채널을 보고 관리할 수 있습니다.

1. W&B에 로그인하고 **팀 설정**으로 이동합니다.
2. **Slack 채널 통합** 섹션에서 각 Slack 대상을 봅니다.
3. 휴지통 아이콘을 클릭하여 대상을 삭제합니다.

## 자동화 생성
[W&B 팀을 Slack에 연결]({{< relref path="#connect-to-slack" lang="ko" >}})한 후 **Registry** 또는 **Project**를 선택한 다음 다음 단계에 따라 Slack 채널에 알림을 보내는 자동화를 만듭니다.

{{< tabpane text=true >}}
{{% tab "Registry" %}}
Registry 관리자는 해당 Registry에서 자동화를 생성할 수 있습니다.

1. W&B에 로그인합니다.
2. Registry 이름을 클릭하여 세부 정보를 봅니다.
3. Registry 범위로 자동화를 생성하려면 **Automations** 탭을 클릭한 다음 **자동화 생성**을 클릭합니다. Registry 범위로 지정된 자동화는 해당 컬렉션(향후 생성된 컬렉션 포함) 모두에 자동으로 적용됩니다.

    Registry에서 특정 컬렉션에만 범위가 지정된 자동화를 생성하려면 컬렉션 작업 `...` 메뉴를 클릭한 다음 **자동화 생성**을 클릭합니다. 또는 컬렉션을 보는 동안 컬렉션 세부 정보 페이지의 **Automations** 섹션에서 **자동화 생성** 버튼을 사용하여 컬렉션에 대한 자동화를 만듭니다.
4. 감시할 [**Event**]({{< relref path="/guides/core/automations/automation-events.md" lang="ko" >}})를 선택합니다.

    이벤트에 따라 나타나는 추가 필드를 작성합니다. 예를 들어 **아티팩트 에일리어스가 추가됨**을 선택한 경우 **에일리어스 정규식**을 지정해야 합니다.
    
    **다음 단계**를 클릭합니다.
5. [Slack 통합]({{< relref path="#add-a-slack-integration" lang="ko" >}})을 소유한 팀을 선택합니다.
6. **액션 유형**을 **Slack 알림**으로 설정합니다. Slack 채널을 선택한 다음 **다음 단계**를 클릭합니다.
7. 자동화 이름을 입력합니다. 선택적으로 설명을 제공합니다.
8. **자동화 생성**을 클릭합니다.

{{% /tab %}}
{{% tab "Project" %}}
W&B 관리자는 프로젝트에서 자동화를 생성할 수 있습니다.

1. W&B에 로그인합니다.
2. 프로젝트 페이지로 이동하여 **Automations** 탭을 클릭합니다.
3. **자동화 생성**을 클릭합니다.
4. 감시할 [**Event**]({{< relref path="/guides/core/automations/automation-events.md" lang="ko" >}})를 선택합니다.

    이벤트에 따라 나타나는 추가 필드를 작성합니다. 예를 들어 **아티팩트 에일리어스가 추가됨**을 선택한 경우 **에일리어스 정규식**을 지정해야 합니다.
    
    **다음 단계**를 클릭합니다.
5. [Slack 통합]({{< relref path="#add-a-slack-integration" lang="ko" >}})을 소유한 팀을 선택합니다.
6. **액션 유형**을 **Slack 알림**으로 설정합니다. Slack 채널을 선택한 다음 **다음 단계**를 클릭합니다.
7. 자동화 이름을 입력합니다. 선택적으로 설명을 제공합니다.
8. **자동화 생성**을 클릭합니다.

{{% /tab %}}
{{< /tabpane >}}

## 자동화 보기 및 관리

{{< tabpane text=true >}}
{{% tab "Registry" %}}

- Registry의 **Automations** 탭에서 Registry의 자동화를 관리합니다.
- 컬렉션의 세부 정보 페이지의 **Automations** 섹션에서 컬렉션의 자동화를 관리합니다.

이러한 페이지에서 Registry 관리자는 기존 자동화를 관리할 수 있습니다.
- 자동화의 세부 정보를 보려면 해당 이름을 클릭합니다.
- 자동화를 편집하려면 해당 작업 `...` 메뉴를 클릭한 다음 **자동화 편집**을 클릭합니다.
- 자동화를 삭제하려면 해당 작업 `...` 메뉴를 클릭한 다음 **자동화 삭제**를 클릭합니다. 확인이 필요합니다.

{{% /tab %}}
{{% tab "Project" %}}
W&B 관리자는 프로젝트의 **Automations** 탭에서 프로젝트의 자동화를 보고 관리할 수 있습니다.

- 자동화의 세부 정보를 보려면 해당 이름을 클릭합니다.
- 자동화를 편집하려면 해당 작업 `...` 메뉴를 클릭한 다음 **자동화 편집**을 클릭합니다.
- 자동화를 삭제하려면 해당 작업 `...` 메뉴를 클릭한 다음 **자동화 삭제**를 클릭합니다. 확인이 필요합니다.
{{% /tab %}}
{{< /tabpane >}}

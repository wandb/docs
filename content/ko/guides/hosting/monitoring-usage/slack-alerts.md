---
title: Configure Slack alerts
menu:
  default:
    identifier: ko-guides-hosting-monitoring-usage-slack-alerts
    parent: monitoring-and-usage
---

W&B 서버를 [Slack](https://slack.com/) 과 통합합니다.

{{% alert %}}
[W&B 전용 클라우드 배포에서 Slack 알림을 설정하는 방법을 보여주는 비디오](https://www.youtube.com/watch?v=JmvKb-7u-oU) (6분)를 시청하세요.
{{% /alert %}}

## Slack 애플리케이션 만들기

아래 절차에 따라 Slack 애플리케이션을 만드세요.

1. https://api.slack.com/apps 를 방문하여 **Create an App** 을 선택합니다.

    {{< img src="/images/hosting/create_an_app.png" alt="" >}}

2. **App Name** 필드에 앱 이름을 입력합니다.
3. 앱을 개발할 Slack 워크스페이스를 선택합니다. 사용하는 Slack 워크스페이스가 알림에 사용하려는 워크스페이스와 동일한지 확인하세요.

    {{< img src="/images/hosting/name_app_workspace.png" alt="" >}}

## Slack 애플리케이션 구성

1. 왼쪽 사이드바에서 **OAth & Permissions** 를 선택합니다.

    {{< img src="/images/hosting/add_an_oath.png" alt="" >}}

2. Scopes 섹션에서 봇에 **incoming_webhook** 스코프를 제공합니다. 스코프는 앱에 개발 워크스페이스에서 작업을 수행할 수 있는 권한을 부여합니다.

    봇에 대한 OAuth 스코프에 대한 자세한 내용은 Slack API 설명서의 봇에 대한 OAuth 스코프 이해 튜토리얼을 참조하세요.

    {{< img src="/images/hosting/save_urls.png" alt="" >}}

3. 리디렉션 URL이 W&B 설치를 가리키도록 구성합니다. 로컬 시스템 설정에서 호스트 URL이 설정된 URL과 동일한 URL을 사용합니다. 인스턴스에 대한 DNS 매핑이 다른 경우 여러 URL을 지정할 수 있습니다.

    {{< img src="/images/hosting/redirect_urls.png" alt="" >}}

4. **Save URLs** 를 선택합니다.
5. 선택적으로 **Restrict API Token Usage** 아래에 IP 범위를 지정하여 W&B 인스턴스의 IP 또는 IP 범위를 허용 목록에 추가할 수 있습니다. 허용된 IP 어드레스를 제한하면 Slack 애플리케이션을 더욱 안전하게 보호할 수 있습니다.

## W&B에 Slack 애플리케이션 등록

1. 배포에 따라 W&B 인스턴스의 **System Settings** 또는 **System Console** 페이지로 이동합니다.

2. 현재 있는 시스템 페이지에 따라 아래 옵션 중 하나를 따르세요.

    - **System Console** 에 있는 경우 **Settings** 로 이동한 다음 **Notifications** 로 이동합니다.

      {{< img src="/images/hosting/register_slack_app_console.png" alt="" >}}

    - **System Settings** 에 있는 경우 **Enable a custom Slack application to dispatch alerts** 를 전환하여 사용자 지정 Slack 애플리케이션을 활성화합니다.

      {{< img src="/images/hosting/register_slack_app.png" alt="" >}}

3. **Slack client ID** 와 **Slack secret** 을 제공한 다음 **Save** 를 클릭합니다. Settings 의 Basic Information 으로 이동하여 애플리케이션의 클라이언트 ID와 secret 을 찾습니다.

4. W&B 앱에서 Slack 인테그레이션을 설정하여 모든 것이 작동하는지 확인합니다.

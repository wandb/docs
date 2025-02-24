---
title: Configure Slack alerts
menu:
  default:
    identifier: ko-guides-hosting-monitoring-usage-slack-alerts
    parent: monitoring-and-usage
---

W&B 서버 를 [Slack](https://slack.com/) 과 통합합니다.

## Slack 애플리케이션 만들기

다음 절차에 따라 Slack 애플리케이션을 만듭니다.

1. https://api.slack.com/apps 를 방문하여 **Create an App** 을 선택합니다.

    {{< img src="/images/hosting/create_an_app.png" alt="" >}}

2. **App Name** 필드에 앱 이름을 입력합니다.
3. 앱을 개발할 Slack workspace 를 선택합니다. 사용하는 Slack workspace 가 알림에 사용할 workspace 와 동일한지 확인하십시오.

    {{< img src="/images/hosting/name_app_workspace.png" alt="" >}}

## Slack 애플리케이션 구성

1. 왼쪽 사이드바에서 **OAth & Permissions** 를 선택합니다.

    {{< img src="/images/hosting/add_an_oath.png" alt="" >}}

2. Scopes 섹션에서 봇에 **incoming_webhook** 스코프를 제공합니다. 스코프는 앱이 개발 workspace 에서 작업을 수행할 수 있는 권한을 부여합니다.

    봇에 대한 OAuth 스코프에 대한 자세한 내용은 Slack API 문서에서 봇에 대한 OAuth 스코프 이해 튜토리얼을 참조하십시오.

    {{< img src="/images/hosting/save_urls.png" alt="" >}}

3. 리디렉션 URL이 W&B 설치를 가리키도록 구성합니다. 로컬 시스템 설정에서 호스트 URL이 설정된 URL과 동일한 URL을 사용하십시오. 인스턴스에 대한 DNS 매핑이 다른 경우 여러 URL을 지정할 수 있습니다.

    {{< img src="/images/hosting/redirect_urls.png" alt="" >}}

4. **Save URLs** 를 선택합니다.
5. 필요에 따라 **Restrict API Token Usage** 에서 IP 범위를 지정하여 W&B 인스턴스의 IP 또는 IP 범위를 허용 목록에 추가합니다. 허용된 IP 어드레스를 제한하면 Slack 애플리케이션을 더욱 안전하게 보호할 수 있습니다.

## W&B 에 Slack 애플리케이션 등록

1. 배포에 따라 W&B 인스턴스의 **System Settings** 또는 **System Console** 페이지로 이동합니다.

2. 현재 있는 시스템 페이지에 따라 다음 옵션 중 하나를 따르십시오.

    - **System Console** 에 있는 경우 **Settings** 로 이동한 다음 **Notifications** 로 이동합니다.

      {{< img src="/images/hosting/register_slack_app_console.png" alt="" >}}

    - **System Settings** 에 있는 경우 **Enable a custom Slack application to dispatch alerts** 를 토글하여 사용자 지정 Slack 애플리케이션을 활성화합니다.

      {{< img src="/images/hosting/register_slack_app.png" alt="" >}}

3. **Slack client ID** 와 **Slack secret** 를 제공한 다음 **Save** 를 클릭합니다. Settings 의 Basic Information 으로 이동하여 애플리케이션의 클라이언트 ID 와 secret 를 찾습니다.

4. W&B 앱에서 Slack integration 을 설정하여 모든 것이 작동하는지 확인합니다.

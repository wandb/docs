---
title: Slack 알림 설정
menu:
  default:
    identifier: ko-guides-hosting-monitoring-usage-slack-alerts
    parent: monitoring-and-usage
---

W&B 서버를 [Slack](https://slack.com/)과 연동하세요.
{{% alert %}}
[W&B 전용 클라우드 배포에서 Slack 알림을 설정하는 방법을 시연하는 영상](https://www.youtube.com/watch?v=JmvKb-7u-oU) 보기 (6분).
{{% /alert %}}

## Slack 애플리케이션 만들기

아래 절차에 따라 Slack 애플리케이션을 생성하세요.

1. https://api.slack.com/apps 에 접속한 후 **Create an App**을 선택하세요.

    {{< img src="/images/hosting/create_an_app.png" alt="Create an App 버튼" >}}

2. **App Name** 필드에 앱 이름을 입력하세요.
3. 앱을 개발할 Slack 워크스페이스를 선택하세요. 알림에 사용할 워크스페이스와 동일한 워크스페이스를 사용해야 합니다.

    {{< img src="/images/hosting/name_app_workspace.png" alt="앱 이름 및 워크스페이스 선택" >}}

## Slack 애플리케이션 설정

1. 왼쪽 사이드바에서 **OAth & Permissions**를 선택하세요.

    {{< img src="/images/hosting/add_an_oath.png" alt="OAuth & Permissions 메뉴" >}}

2. Scopes 섹션에서 봇에 **incoming_webhook** 스코프를 부여하세요. 스코프는 앱이 개발 워크스페이스에서 작업을 수행할 수 있는 권한을 부여합니다.

    OAuth 스코프에 대한 자세한 내용은 Slack API 문서의 'Understanding OAuth scopes for Bots' 튜토리얼을 참고하세요.

    {{< img src="/images/hosting/save_urls.png" alt="Bot 토큰 스코프" >}}

3. Redirect URL을 W&B 인스턴스의 주소로 설정하세요. 로컬 시스템 설정에 구성된 호스트 URL과 동일한 URL을 사용해야 합니다. 인스턴스에 여러 DNS 매핑이 있을 경우 여러 URL을 지정할 수 있습니다.

    {{< img src="/images/hosting/redirect_urls.png" alt="Redirect URL 설정" >}}

4. **Save URLs**를 선택하세요.
5. 필요에 따라 **Restrict API Token Usage**에서 IP 또는 IP 범위를 허용 목록에 추가하여 W&B 인스턴스의 IP 대역을 지정할 수 있습니다. 허용된 IP를 제한하면 Slack 애플리케이션을 더욱 안전하게 보호할 수 있습니다.

## W&B에 Slack 애플리케이션 등록

1. W&B 인스턴스의 **System Settings** 또는 **System Console** 페이지로 이동하세요(배포 환경에 따라 다름).

2. 접근한 시스템 페이지에 따라 아래 옵션 중 하나를 따르세요:

    - **System Console**에 있는 경우: **Settings**로 이동한 뒤 **Notifications** 메뉴로 이동하세요.

      {{< img src="/images/hosting/register_slack_app_console.png" alt="System Console 알림" >}}

    - **System Settings**에 있는 경우: **Enable a custom Slack application to dispatch alerts** 토글을 활성화하세요.

      {{< img src="/images/hosting/register_slack_app.png" alt="Slack 애플리케이션 활성화 토글" >}}

3. **Slack client ID**와 **Slack secret**을 입력한 뒤 **Save**를 클릭하세요. 애플리케이션의 client ID와 secret은 Settings의 Basic Information에서 확인할 수 있습니다.

4. W&B 애플리케이션에서 Slack 인테그레이션을 설정해 정상 작동하는지 확인하세요.
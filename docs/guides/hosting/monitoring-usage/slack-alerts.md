---
title: Configure Slack alerts
displayed_sidebar: default
---

Integrate W&B Server with [Slack](https://slack.com/).

## Slack 애플리케이션 생성

아래 절차를 따라 Slack 애플리케이션을 생성하세요.

1. https://api.slack.com/apps 를 방문하여 **Create an App**을 선택합니다.

![](/images/hosting/create_an_app.png)

2. **App Name** 필드에 앱 이름을 입력합니다.
3. 앱을 개발하려는 Slack 워크스페이스를 선택합니다. 사용하려는 Slack 워크스페이스가 경고에 사용할 워크스페이스와 동일한지 확인하세요.

![](/images/hosting/name_app_workspace.png)

## Slack 애플리케이션 설정

1. 왼쪽 사이드바에서 **OAth & Permissions**를 선택합니다.

![](/images/hosting/add_an_oath.png)

2. Scopes 섹션 내에서 봇에게 **incoming_webhook** 스코프를 제공합니다. 스코프는 개발 워크스페이스에서 앱이 작업을 수행할 수 있는 권한을 부여합니다.

   봇을 위한 OAuth 스코프에 대한 추가 정보는 Slack api 문서의 Understanding OAuth scopes for Bots 튜토리얼을 참조하세요.

![](/images/hosting/save_urls.png)

3. Redirect URL을 W&B 설치 지점으로 설정합니다. 로컬 시스템 설정에서 호스트 URL로 설정된 것과 동일한 URL을 사용하세요. 인스턴스에 대한 서로 다른 DNS 매핑이 있는 경우 여러 URL을 지정할 수 있습니다.

![](/images/hosting/redirect_urls.png)

4. **Save URLs**를 선택합니다.
5. 선택적으로 **Restrict API Token Usage** 아래에서 IP 범위를 지정하여 W&B 인스턴스의 IP 또는 IP 범위를 허용 목록에 추가할 수 있습니다. 허용된 IP 주소를 제한하여 Slack 애플리케이션을 더욱 안전하게 보호할 수 있습니다.

## Slack 애플리케이션을 W&B에 등록

1. 배포에 따라 W&B 인스턴스의 **System Settings** 또는 **System Console** 페이지로 이동합니다.

2. 있는 시스템 페이지에 따라 아래 옵션 중 하나를 따르세요:

- **System Console**에 있는 경우: **Settings**로 이동한 후 **Notifications**로 이동합니다.

![](/images/hosting/register_slack_app_console.png)

- **System Settings**에 있는 경우: **Enable a custom Slack application to dispatch alerts**를 토글하여 사용자 정의 Slack 애플리케이션을 활성화합니다.

![](/images/hosting/register_slack_app.png)

3. **Slack client ID**와 **Slack secret**을 입력한 다음 **Save**를 클릭합니다. **Settings**에서 Basic Information으로 이동하여 애플리케이션의 클라이언트 ID와 비밀키를 찾습니다.

4. W&B 앱에서 Slack 인테그레이션을 설정하여 모든 것이 작동하는지 확인하세요.
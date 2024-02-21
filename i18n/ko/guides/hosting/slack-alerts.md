---
displayed_sidebar: default
---

# Slack 알림

W&B 서버를 [Slack](https://slack.com/)과 통합합니다.

## Slack 애플리케이션 생성

아래 절차에 따라 Slack 애플리케이션을 생성하세요.

1. https://api.slack.com/apps를 방문하여 **Create an App**을 선택합니다.

![](/images/hosting/create_an_app.png)

2. **App Name** 필드에 앱의 이름을 입력합니다.
3. 앱을 개발하고자 하는 Slack 워크스페이스를 선택하세요. 알림을 사용할 워크스페이스가 방문한 Slack 워크스페이스와 동일한지 확인하세요.

![](/images/hosting/name_app_workspace.png)

## Slack 애플리케이션 구성

1. 왼쪽 사이드바에서 **OAuth & Permissions**을 선택합니다.

![](/images/hosting/add_an_oath.png)

2. Scopes 섹션에서, 봇에게 **incoming_webhook** 범위를 제공합니다. 범위는 앱이 개발 워크스페이스에서 작업을 수행할 수 있는 권한을 부여합니다.

   봇에 대한 OAuth 범위에 대한 자세한 정보는 Slack api 문서의 Understanding OAuth scopes for Bots 튜토리얼을 참조하세요.

![](/images/hosting/save_urls.png)

3. 리디렉션 URL을 W&B 설치 위치로 설정합니다. 로컬 시스템 설정에서 호스트 URL로 설정된 것과 동일한 URL을 사용하세요. 인스턴스에 대해 다른 DNS 매핑이 있는 경우 여러 URL을 지정할 수 있습니다.

![](/images/hosting/redirect_urls.png)

4. **Save URLs**을 선택합니다.
5. 선택적으로 **Restrict API Token Usage** 아래에 IP 범위를 지정하여, W&B 인스턴스의 IP 또는 IP 범위를 허용 목록에 추가할 수 있습니다. 허용된 IP 주소를 제한하면 Slack 애플리케이션의 보안을 더욱 강화할 수 있습니다.

## W&B에 Slack 애플리케이션 등록

1. W&B 인스턴스의 **System Settings** 페이지로 이동하여 **Enable a custom Slack application to dispatch alerts**를 활성화하여 사용자 지정 Slack 애플리케이션을 사용합니다:

![](/images/hosting/register_slack_app.png)

Slack 애플리케이션의 클라이언트 ID와 비밀번호를 제공해야 합니다. 애플리케이션의 클라이언트 ID와 비밀번호를 찾으려면 설정에서 기본 정보로 이동하세요.

2. W&B 앱에서 Slack 통합을 설정하여 모든 것이 작동하는지 확인합니다.
---
displayed_sidebar: default
---

# Slack 알림

W&B 서버를 [Slack](https://slack.com/)과 통합합니다.

## Slack 애플리케이션 생성하기

아래 절차에 따라 Slack 애플리케이션을 생성하세요.

1. https://api.slack.com/apps 에 방문하여 **앱 생성**을 선택하세요.

![](/images/hosting/create_an_app.png)

2. **앱 이름** 필드에 앱의 이름을 제공하세요.
3. 앱을 개발하고자 하는 Slack 워크스페이스를 선택하세요. 알림을 사용하고자 하는 워크스페이스가 선택한 Slack 워크스페이스와 동일한지 확인하세요.

![](/images/hosting/name_app_workspace.png)

## Slack 애플리케이션 구성하기

1. 왼쪽 사이드바에서 **OAth & 권한**을 선택하세요.

![](/images/hosting/add_an_oath.png)

2. 범위 섹션 내에서, **incoming_webhook** 범위로 봇에게 권한을 부여하세요. 범위는 앱이 개발 워크스페이스에서 작업을 수행할 수 있는 권한을 부여합니다.

   Bots를 위한 OAuth 범위에 대한 자세한 정보는 Slack api 문서의 Bots를 위한 OAuth 범위 이해하기 튜토리얼을 참조하세요.

![](/images/hosting/save_urls.png)

3. W&B 설치로 리다이렉트 URL을 구성하세요. 로컬 시스템 설정에서 호스트 URL로 설정된 것과 동일한 URL을 사용하세요. 인스턴스에 대한 다른 DNS 매핑이 있는 경우 여러 URL을 지정할 수 있습니다.

![](/images/hosting/redirect_urls.png)

4. **URL 저장**을 선택하세요.
5. 선택적으로 **API 토큰 사용 제한** 아래에서 IP 범위를 지정하여, W&B 인스턴스의 IP 또는 IP 범위를 허용 목록에 추가할 수 있습니다. 허용된 IP 어드레스를 제한하면 Slack 애플리케이션의 보안을 더욱 강화할 수 있습니다.

## W&B에 Slack 애플리케이션 등록하기

1. W&B 인스턴스의 **시스템 설정** 페이지로 이동하세요. **사용자 지정 Slack 애플리케이션을 사용하여 알림을 전송할 수 있도록 설정**을 활성화하여 사용자 지정 Slack 애플리케이션을 활성화하세요:

![](/images/hosting/register_slack_app.png)

Slack 애플리케이션의 클라이언트 ID와 비밀번호를 제공해야 합니다. 애플리케이션의 클라이언트 ID와 비밀번호를 찾으려면 설정에서 기본 정보로 이동하세요.

2. W&B 앱에서 Slack 인테그레이션을 설정하여 모든 것이 정상적으로 작동하는지 확인하세요.
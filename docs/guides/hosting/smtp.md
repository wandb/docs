---
displayed_sidebar: default
---

# SMTP 구성

W&B 서버에서 인스턴스나 팀에 사용자를 추가하면 이메일 초대장이 발송됩니다. 이러한 이메일 초대장을 발송하기 위해 W&B는 타사 메일 서버를 사용합니다. 경우에 따라 조직은 회사 네트워크를 벗어나는 트래픽에 대해 엄격한 정책을 가지고 있어 이러한 이메일 초대장이 최종 사용자에게 전송되지 않을 수 있습니다. W&B 서버는 내부 SMTP 서버를 통해 이러한 초대 이메일을 발송하도록 구성하는 옵션을 제공합니다.

구성하려면 아래 단계를 따르세요:

- docker 컨테이너 또는 kubernetes 배포에서 `GORILLA_EMAIL_SINK` 환경 변수를 `smtp://<user:password>@smtp.host.com:<port>`로 설정하세요.
- `username`과 `password`는 선택 사항입니다.
- 인증이 필요 없는 SMTP 서버를 사용하는 경우 환경 변수의 값으로 `GORILLA_EMAIL_SINK=smtp://smtp.host.com:<port>`와 같이 설정하면 됩니다.
- SMTP에 대해 자주 사용되는 포트 번호는 587, 465 및 25입니다. 이는 사용하는 메일 서버의 유형에 따라 다를 수 있습니다.
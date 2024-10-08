---
title: Configure SMTP
displayed_sidebar: default
---

W&B 서버에서 인스턴스나 팀에 사용자를 추가하면 이메일 초대가 발송됩니다. 이 이메일 초대를 보내기 위해 W&B는 제 3자 메일 서버를 사용합니다. 경우에 따라 조직에서는 기업 네트워크를 떠나는 트래픽에 대한 엄격한 정책을 가지고 있어 이러한 이메일 초대가 최종 사용자에게 절대 전달되지 않을 수 있습니다. W&B 서버는 내부 SMTP 서버를 통해 이 초대 이메일을 전송하도록 구성할 수 있는 옵션을 제공합니다.

구성하려면 아래 단계를 따르세요:

- docker 컨테이너 또는 kubernetes 배포의 환경 변수 `GORILLA_EMAIL_SINK`를 `smtp://<user:password>@smtp.host.com:<port>`로 설정합니다.
- `username`과 `password`는 선택 사항입니다.
- 인증되지 않은 것으로 설계된 SMTP 서버를 사용하는 경우 `GORILLA_EMAIL_SINK=smtp://smtp.host.com:<port>`와 같이 환경 변수의 값을 설정하면 됩니다.
- SMTP에서 일반적으로 사용되는 포트 번호는 포트 587, 465 및 25입니다. 이는 사용 중인 메일 서버의 유형에 따라 다를 수 있습니다.
- 초기에는 `noreply@wandb.com`으로 설정된 SMTP의 기본 발신 이메일 주소를 구성하려면 서버의 환경 변수 `GORILLA_EMAIL_FROM_ADDRESS`를 원하는 발신 이메일 주소로 설정하여 업데이트할 수 있습니다.
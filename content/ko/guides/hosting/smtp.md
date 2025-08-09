---
title: SMTP 구성
menu:
  default:
    identifier: ko-guides-hosting-smtp
    parent: w-b-platform
weight: 6
---

W&B 서버에서는 인스턴스나 팀에 사용자를 추가하면 이메일 초대장이 발송됩니다. 이러한 이메일 초대를 보내기 위해 W&B는 외부 메일 서버를 사용합니다. 경우에 따라 조직에서 사내 네트워크를 벗어나는 트래픽에 대해 엄격한 정책을 적용하는 경우가 있어, 이메일 초대장이 최종 사용자에게 전달되지 않을 수 있습니다. W&B 서버에서는 내부 SMTP 서버를 통해 이러한 초대 이메일을 발송하도록 설정할 수 있는 옵션을 제공합니다.

설정 방법은 아래 단계를 따르세요:

- docker 컨테이너 또는 kubernetes 배포 환경에서 `GORILLA_EMAIL_SINK` 환경 변수에 `smtp://<user:password>@smtp.host.com:<port>` 값을 지정합니다.
- `username`과 `password`는 선택 사항입니다.
- 인증이 필요 없는 SMTP 서버를 사용하는 경우 환경 변수 값을 `GORILLA_EMAIL_SINK=smtp://smtp.host.com:<port>`와 같이 설정하면 됩니다.
- SMTP에서 일반적으로 사용되는 포트 번호는 587, 465, 25번이지만, 사용하는 메일 서버 종류에 따라 다를 수 있습니다.
- SMTP의 기본 발신자 이메일 어드레스는 최초에는 `noreply@wandb.com`으로 설정되어 있으나, 원하는 이메일 어드레스로 변경할 수 있습니다. 이 경우 서버에서 `GORILLA_EMAIL_FROM_ADDRESS` 환경 변수에 원하는 발신자 이메일 어드레스를 지정하면 반영됩니다.
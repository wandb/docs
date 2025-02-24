---
title: Configure SMTP
menu:
  default:
    identifier: ko-guides-hosting-smtp
    parent: w-b-platform
weight: 6
---

W&B server에서 인스턴스 또는 Team에 Users를 추가하면 이메일 초대가 실행됩니다. 이러한 이메일 초대를 보내기 위해 W&B는 타사 메일 서버를 사용합니다. 경우에 따라 조직은 기업 네트워크를 벗어나는 트래픽에 대해 엄격한 정책을 적용하여 이러한 이메일 초대가 최종 User에게 전송되지 않을 수 있습니다. W&B server는 내부 SMTP 서버를 통해 이러한 초대 이메일을 보내도록 구성하는 옵션을 제공합니다.

구성하려면 다음 단계를 따르세요.

- docker 컨테이너 또는 kubernetes 배포에서 `GORILLA_EMAIL_SINK` 환경 변수를 `smtp://<user:password>@smtp.host.com:<port>`로 설정합니다.
- `username`과 `password`는 선택 사항입니다.
- 인증되지 않도록 설계된 SMTP 서버를 사용하는 경우 `GORILLA_EMAIL_SINK=smtp://smtp.host.com:<port>`와 같이 환경 변수의 값을 설정하기만 하면 됩니다.
- SMTP에 일반적으로 사용되는 포트 번호는 587, 465, 25번 포트입니다. 이는 사용 중인 메일 서버 유형에 따라 다를 수 있습니다.
- SMTP의 기본 발신자 이메일 어드레스를 구성하려면(초기에는 `noreply@wandb.com`으로 설정됨), 서버에서 `GORILLA_EMAIL_FROM_ADDRESS` 환경 변수를 원하는 발신자 이메일 어드레스로 업데이트하면 됩니다.
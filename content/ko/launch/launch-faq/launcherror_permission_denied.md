---
title: How do I fix a "permission denied" error in Launch?
menu:
  launch:
    identifier: ko-launch-launch-faq-launcherror_permission_denied
    parent: launch-faq
---

`Launch Error: Permission denied` 오류 메시지가 발생하면 원하는 프로젝트에 로그할 권한이 부족하다는 의미입니다. 가능한 원인은 다음과 같습니다.

1. 이 머신에 로그인하지 않았습니다. 커맨드라인에서 [`wandb login`]({{< relref path="/ref/cli/wandb-login.md" lang="ko" >}}) 을 실행하세요.
2. 지정된 엔티티가 존재하지 않습니다. 엔티티는 사용자 이름 또는 기존 팀 이름이어야 합니다. 필요한 경우 [Subscriptions page](https://app.wandb.ai/billing)에서 팀을 만드세요.
3. 프로젝트 권한이 없습니다. 프로젝트 생성자에게 프로젝트에 run을 로그할 수 있도록 개인 정보 보호 설정을 **Open**으로 변경하도록 요청하세요.

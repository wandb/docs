---
title: Launch 에서 "permission denied" 오류를 어떻게 해결하나요?
menu:
  launch:
    identifier: ko-launch-launch-faq-launcherror_permission_denied
    parent: launch-faq
---

`Launch Error: Permission denied` 에러 메시지가 발생하면, 원하는 프로젝트에 로그를 남길 권한이 부족하다는 의미입니다. 가능한 원인은 다음과 같습니다:

1. 현재 이 머신에 로그인되어 있지 않습니다. 커맨드라인에서 [`wandb login`]({{< relref path="/ref/cli/wandb-login.md" lang="ko" >}})을 실행하세요.
2. 지정한 entity가 존재하지 않습니다. entity는 본인의 사용자 이름이나 존재하는 팀의 이름이어야 합니다. 필요하다면 [Subscriptions page](https://app.wandb.ai/billing)에서 팀을 생성하세요.
3. 프로젝트에 대한 권한이 없습니다. 프로젝트 생성자에게 요청하여 privacy 설정을 **Open**으로 변경하면 run을 해당 프로젝트에 로그할 수 있습니다.
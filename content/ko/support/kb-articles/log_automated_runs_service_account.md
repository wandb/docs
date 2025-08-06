---
title: 지속적 인테그레이션이나 내부 툴에서 실행된 run 을 어떻게 로그할 수 있나요?
menu:
  support:
    identifier: ko-support-kb-articles-log_automated_runs_service_account
support:
- run
- 로그
toc_hide: true
type: docs
url: /support/:filename
---

자동화된 테스트나 W&B에 로그를 남기는 내부 툴을 실행하려면, 팀 설정 페이지에서 **Service Account**를 생성하세요. 이 작업을 통해 자동화된 작업(예: 지속적 인테그레이션을 통해 실행되는 작업)을 위한 서비스 API 키를 사용할 수 있습니다. 서비스 계정 작업을 특정 사용자에게 할당하려면 `WANDB_USERNAME` 또는 `WANDB_USER_EMAIL` 환경 변수를 설정하세요.

{{< img src="/images/track/common_questions_automate_runs.png" alt="서비스 계정 생성" >}}
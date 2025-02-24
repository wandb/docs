---
title: How do I log runs launched by continuous integration or internal tools?
menu:
  support:
    identifier: ko-support-log_automated_runs_service_account
tags:
- runs
- logs
toc_hide: true
type: docs
---

W&B에 로그하는 자동화된 테스트 또는 내부 툴을 실행하려면 팀 설정 페이지에서 **서비스 계정**을 생성하세요. 이 작업을 통해 지속적인 통합을 통해 실행되는 작업을 포함하여 자동화된 작업에 서비스 API 키를 사용할 수 있습니다. 서비스 계정 작업을 특정 사용자에 귀속시키려면 `WANDB_USERNAME` 또는 `WANDB_USER_EMAIL` 환경 변수를 설정하세요.

{{< img src="/images/track/common_questions_automate_runs.png" alt="자동화된 작업을 위해 팀 설정 페이지에서 서비스 계정을 만드세요." >}}

---
title: How can I resolve the Filestream rate limit exceeded error?
menu:
  support:
    identifier: ko-support-kb-articles-filestream_rate_limit_exceeded_error
support:
- connectivity
- outage
toc_hide: true
type: docs
url: /ko/support/:filename
---

Weights & Biases (W&B)에서 "Filestream 속도 제한 초과" 오류를 해결하려면 다음 단계를 따르세요.

**로깅 최적화**:
  - 로깅 빈도를 줄이거나 로그를 일괄 처리하여 API 요청을 줄입니다.
  - 실험 시작 시간을 분산시켜 동시 API 요청을 피합니다.

**서비스 중단 확인**:
  - [W&B 상태 업데이트](https://status.wandb.com)를 확인하여 문제가 일시적인 서버 측 문제에서 발생하지 않았는지 확인합니다.

**지원팀에 문의**:
  - 실험 설정에 대한 자세한 내용을 첨부하여 W&B 지원팀(support@wandb.com)에 문의하여 속도 제한 증가를 요청합니다.

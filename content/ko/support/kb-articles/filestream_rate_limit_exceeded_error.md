---
title: Filestream 속도 제한 초과 오류를 어떻게 해결할 수 있나요?
menu:
  support:
    identifier: ko-support-kb-articles-filestream_rate_limit_exceeded_error
support:
- 연결성
- 장애
toc_hide: true
type: docs
url: /support/:filename
---

W&B에서 "Filestream rate limit exceeded" 오류를 해결하려면 다음 단계를 따라주세요:

**로그 최적화**:
  - 로그 빈도를 줄이거나 로그를 배치로 전송하여 API 요청 횟수를 감소시키세요.
  - 여러 실험의 시작 시간을 서로 다르게 조정해 동시에 발생하는 API 요청을 피하세요.

**장애 여부 확인**:
  - [W&B 상태 페이지](https://status.wandb.com)에서 임시 서버 측 문제로 인한 오류가 아닌지 확인하세요.

**고객 지원 문의**:
  - 실험 환경에 대한 상세 정보를 첨부해 W&B 지원팀(support@wandb.com)에 연락하여 할당량 상향을 요청하세요.
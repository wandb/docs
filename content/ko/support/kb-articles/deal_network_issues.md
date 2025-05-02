---
title: How do I deal with network issues?
menu:
  support:
    identifier: ko-support-kb-articles-deal_network_issues
support:
- connectivity
toc_hide: true
type: docs
url: /ko/support/:filename
translationKey: deal_network_issues
---
SSL 또는 네트워크 오류가 발생하는 경우(예: `wandb: Network error (ConnectionError), entering retry loop`), 다음 해결 방법을 사용하세요.

1. SSL 인증서를 업그레이드하세요. Ubuntu 서버에서 `update-ca-certificates`를 실행합니다. 유효한 SSL 인증서는 보안 위험을 완화하기 위해 트레이닝 로그를 동기화하는 데 필수적입니다.
2. 네트워크 연결이 불안정한 경우, [선택적 환경 변수]({{< relref path="/guides/models/track/environment-variables.md#optional-environment-variables" lang="ko" >}}) `WANDB_MODE`를 `offline`으로 설정하여 오프라인 모드로 작동하고, 인터넷 엑세스가 가능한 장치에서 나중에 파일을 동기화합니다.
3. 로컬에서 실행되고 클라우드 서버와 동기화를 피하는 [W&B Private Hosting]({{< relref path="/guides/hosting/" lang="ko" >}}) 사용을 고려하세요.

`SSL CERTIFICATE_VERIFY_FAILED` 오류의 경우, 이 문제는 회사 방화벽에서 발생할 수 있습니다. 로컬 CA를 구성하고 다음을 실행합니다.

`export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt`

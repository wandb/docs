---
title: 네트워크 문제를 어떻게 해결할 수 있나요?
menu:
  support:
    identifier: ko-support-kb-articles-deal_network_issues
support:
- 연결성
toc_hide: true
type: docs
url: /support/:filename
---

SSL 또는 네트워크 오류가 발생하는 경우(예: `wandb: Network error (ConnectionError), entering retry loop`), 다음과 같은 방법으로 해결할 수 있습니다.

1. SSL 인증서를 업그레이드하세요. Ubuntu 서버에서는 `update-ca-certificates` 명령어를 실행하면 됩니다. 유효한 SSL 인증서는 트레이닝 로그를 동기화할 때 보안 위험을 줄이기 위해 필수입니다.
2. 네트워크 연결이 불안정할 경우, [옵션 환경 변수]({{< relref path="/guides/models/track/environment-variables.md#optional-environment-variables" lang="ko" >}}) `WANDB_MODE`를 `offline`으로 설정하여 오프라인 모드로 동작시키고, 나중에 인터넷이 연결된 디바이스에서 파일을 동기화하세요.
3. [W&B Private Hosting]({{< relref path="/guides/hosting/" lang="ko" >}})을 고려해보세요. 이 방식은 로컬에서 실행되며, 클라우드 서버로의 동기화를 피할 수 있습니다.

`SSL CERTIFICATE_VERIFY_FAILED` 오류의 경우, 회사 방화벽에서 비롯될 수 있습니다. 로컬 CA를 설정하고 아래 명령어를 실행하세요.

`export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt`
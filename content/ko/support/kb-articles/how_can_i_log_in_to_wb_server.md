---
title: W&B 서버에 어떻게 로그인할 수 있나요?
menu:
  support:
    identifier: ko-support-kb-articles-how_can_i_log_in_to_wb_server
support:
- user 관리
toc_hide: true
type: docs
url: /support/:filename
---

로그인 URL을 설정하는 방법은 다음 두 가지 중 하나입니다:

- [환경 변수]({{< relref path="guides/models/track/environment-variables.md" lang="ko" >}}) `WANDB_BASE_URL`를 Server URL로 설정합니다.
- [`wandb login`]({{< relref path="/ref/cli/wandb-login.md" lang="ko" >}})의 `--host` 플래그를 Server URL로 설정합니다.
---
title: How can I log in to W&B Server?
menu:
  support:
    identifier: ko-support-how_can_i_log_in_to_wb_server
tags:
- user management
toc_hide: true
type: docs
---

다음 방법 중 하나를 사용하여 로그인 URL을 설정하세요.

- [환경 변수]({{< relref path="guides/models/track/environment-variables.md" lang="ko" >}}) `WANDB_BASE_URL`을 Server URL로 설정합니다.
- [`wandb login`]({{< relref path="/ref/cli/wandb-login.md" lang="ko" >}})의 `--host` 플래그를 Server URL로 설정합니다.

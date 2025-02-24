---
title: What are features that are not available to anonymous users?
menu:
  support:
    identifier: ko-support-anon_users_unavailable_features
tags:
- anonymous
toc_hide: true
type: docs
---

* **지속적인 데이터 없음**: Run은 익명 계정으로 7일 동안 저장됩니다. 실제 계정에 저장하여 익명 Run 데이터를 클레임하세요.

{{< img src="/images/app_ui/anon_mode_no_data.png" alt="" >}}

* **아티팩트 로깅 없음**: 익명 Run에 아티팩트 를 로그하려고 하면 커맨드라인 에 경고가 표시됩니다.
    ```bash
    wandb: WARNING Artifacts logged anonymously cannot be claimed and expire after 7 days.
    ```

* **프로필 또는 설정 페이지 없음**: UI에는 실제 계정에만 유용한 특정 페이지가 포함되어 있지 않습니다.

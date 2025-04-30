---
title: What are features that are not available to anonymous users?
menu:
  support:
    identifier: ko-support-kb-articles-anon_users_unavailable_features
support:
- anonymous
toc_hide: true
type: docs
url: /ko/support/:filename
---

* **지속적인 데이터 없음**: Run은 익명 계정에서 7일 동안 저장됩니다. 실제 계정에 저장하여 익명 Run 데이터를 클레임하세요.

{{< img src="/images/app_ui/anon_mode_no_data.png" alt="" >}}

* **아티팩트 로깅 없음**: 익명 Run에 아티팩트를 로깅하려고 시도하면 커맨드 라인에 경고가 표시됩니다.
    ```bash
    wandb: WARNING 익명으로 기록된 Artifacts는 클레임할 수 없으며 7일 후에 만료됩니다.
    ```

* **프로필 또는 설정 페이지 없음**: UI에는 특정 페이지가 포함되어 있지 않습니다. 실제 계정에서만 유용하기 때문입니다.

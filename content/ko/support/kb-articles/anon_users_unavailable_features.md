---
title: 익명 사용자는 사용할 수 없는 기능에는 어떤 것이 있나요?
menu:
  support:
    identifier: ko-support-kb-articles-anon_users_unavailable_features
support:
- Anonymous
toc_hide: true
type: docs
url: /support/:filename
---

* **지속적인 데이터 없음**: Run 은 익명 계정에 7일 동안만 저장됩니다. 익명 run 데이터를 실제 계정에 저장하면 소유할 수 있습니다.

{{< img src="/images/app_ui/anon_mode_no_data.png" alt="Anonymous mode interface" >}}

* **artifact 로그 불가**: 익명 run 에 artifact 를 기록하려고 하면 커맨드라인에 경고가 나타납니다:
    ```bash
    wandb: WARNING Artifacts logged anonymously cannot be claimed and expire after 7 days.
    ```

* **프로필 또는 설정 페이지 없음**: UI 에는 실제 계정에만 필요한 일부 페이지가 포함되어 있지 않습니다.
---
title: How do I fix the error `resume='must' but run (<run_id>) doesn't exist`?
menu:
  support:
    identifier: ko-support-kb-articles-how_do_i_fix_the_error_resume_must_but_run_run_id_doesnt_exist
support:
- resuming
- runs
toc_hide: true
type: docs
url: /ko/support/:filename
---

`resume='must'` 오류가 발생했지만 run(<run_id>)이 존재하지 않는 경우, 재개하려는 run이 프로젝트 또는 엔티티 내에 존재하지 않는 것입니다. 올바른 인스턴스에 로그인되어 있고 프로젝트와 엔티티가 설정되어 있는지 확인하세요.

```python
wandb.init(entity=<entity>, project=<project>, id=<run-id>, resume='must')
```

[`wandb login --relogin`]({{< relref path="/ref/cli/wandb-login.md" lang="ko" >}})을 실행하여 인증되었는지 확인하세요.

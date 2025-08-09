---
title: '`resume=''must''`인데 run (<run_id>)가 존재하지 않는다는 오류를 어떻게 해결하나요?'
menu:
  support:
    identifier: ko-support-kb-articles-how_do_i_fix_the_error_resume_must_but_run_run_id_doesnt_exist
support:
- 재개 중
- run
toc_hide: true
type: docs
url: /support/:filename
---

만약 `resume='must' but run (<run_id>) doesn't exist` 오류가 발생한다면, 여러분이 재개하려는 run 이 해당 프로젝트나 entity 내에 존재하지 않는다는 의미입니다. 올바른 인스턴스에 로그인되어 있고, 프로젝트와 entity가 정확하게 설정되어 있는지 확인하세요:

```python
wandb.init(entity=<entity>, project=<project>, id=<run-id>, resume='must')
```

인증 여부를 확인하려면 [`wandb login --relogin`]({{< relref path="/ref/cli/wandb-login.md" lang="ko" >}}) 명령어를 실행하세요.
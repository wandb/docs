---
menu:
  support:
    identifier: ja-support-kb-articles-how_do_i_fix_the_error_resume_must_but_run_run_id_doesnt_exist
support:
- resuming
- runs
title: How do I fix the error `resume='must' but run (<run_id>) doesn't exist`?
toc_hide: true
type: docs
url: /support/:filename
---

If you encounter the error `resume='must' but run (<run_id>) doesn't exist`, the run you are attempting to resume does not exist within the project or entity. Ensure that you are logged in to the correct instance and that the project and entity are set:

```python
wandb.init(entity=<entity>, project=<project>, id=<run-id>, resume='must')
```

Run [`wandb login --relogin`]({{< relref path="/ref/cli/wandb-login.md" lang="ja" >}}) to verify that you are authenticated.
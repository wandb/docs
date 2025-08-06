---
title: run 이름을 run ID로 설정해도 되나요?
menu:
  support:
    identifier: ko-support-kb-articles-just_set_run_name_run_id
support:
- 실험
toc_hide: true
type: docs
url: /support/:filename
---

네. run 이름을 run ID로 덮어쓰려면 다음 코드조각을 사용하세요:

```python
import wandb

with wandb.init() as run:
   # run 이름을 run ID로 설정합니다.
   run.name = run.id
   run.save()
```
---
title: Is it possible to change the group assigned to a run after completion?
menu:
  support:
    identifier: ko-support-kb-articles-change_group_after_completion
support:
- runs
toc_hide: true
type: docs
url: /ko/support/:filename
translationKey: change_group_after_completion
---
API를 사용하여 완료된 run에 할당된 그룹을 변경할 수 있습니다. 이 기능은 웹 UI에 표시되지 않습니다. 다음 코드를 사용하여 그룹을 업데이트하세요.

```python
import wandb

api = wandb.Api()
run = api.run("<ENTITY>/<PROJECT>/<RUN_ID>")
run.group = "NEW-GROUP-NAME"
run.update()
```

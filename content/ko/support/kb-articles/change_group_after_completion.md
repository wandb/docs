---
title: 완료된 run 에 할당된 그룹을 나중에 변경할 수 있나요?
menu:
  support:
    identifier: ko-support-kb-articles-change_group_after_completion
support:
- run
toc_hide: true
type: docs
url: /support/:filename
---

API를 사용하여 완료된 run에 할당된 그룹을 변경할 수 있습니다. 이 기능은 웹 UI에는 표시되지 않습니다. 다음 코드를 사용하여 그룹을 업데이트하세요.

```python
import wandb

api = wandb.Api()
run = api.run("<ENTITY>/<PROJECT>/<RUN_ID>")
run.group = "NEW-GROUP-NAME"
run.update()
```
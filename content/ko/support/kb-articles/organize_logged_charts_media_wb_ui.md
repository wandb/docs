---
title: W&B UI에서 로그된 차트와 미디어를 어떻게 정리할 수 있나요?
menu:
  support:
    identifier: ko-support-kb-articles-organize_logged_charts_media_wb_ui
support:
- 실험
toc_hide: true
type: docs
url: /support/:filename
---

`/` 문자는 W&B UI에서 로그된 패널을 구분하는 데 사용됩니다. 기본적으로 로그된 항목 이름에서 `/` 앞에 오는 부분이 "패널 섹션"이라고 하는 패널 그룹을 정의합니다.

```python
import wandb

with wandb.init() as run:

   run.log({"val/loss": 1.1, "val/acc": 0.3})
   run.log({"train/loss": 0.1, "train/acc": 0.94})
```

[Workspace]({{< relref path="/guides/models/track/project-page.md#workspace-tab" lang="ko" >}}) 설정에서, 패널 그룹을 `/`로 구분된 첫 번째 세그먼트로 할지, 모든 세그먼트로 할지 조정할 수 있습니다.
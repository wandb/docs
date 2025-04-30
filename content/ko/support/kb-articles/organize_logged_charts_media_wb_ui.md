---
title: How can I organize my logged charts and media in the W&B UI?
menu:
  support:
    identifier: ko-support-kb-articles-organize_logged_charts_media_wb_ui
support:
- experiments
toc_hide: true
type: docs
url: /ko/support/:filename
---

`/` 문자는 W&B UI에서 로그된 패널을 구분합니다. 기본적으로 `/` 이전의 로그된 항목 이름 세그먼트는 "패널 섹션"으로 알려진 패널 그룹을 정의합니다.

```python
wandb.log({"val/loss": 1.1, "val/acc": 0.3})
wandb.log({"train/loss": 0.1, "train/acc": 0.94})
```

[Workspace]({{< relref path="/guides/models/track/project-page.md#workspace-tab" lang="ko" >}}) 설정에서 `/`로 구분된 첫 번째 세그먼트 또는 모든 세그먼트를 기반으로 패널 그룹화를 조정합니다.

---
title: Files 탭에 보이지 않는 파일은 어떻게 확인할 수 있나요?
menu:
  support:
    identifier: ko-support-kb-articles-how_can_i_see_files_that_do_not_appear_in_the_files_tab
support:
- Experiments
toc_hide: true
type: docs
url: /support/:filename
---

Files 탭에서는 최대 10,000개의 파일만 표시됩니다. 모든 파일을 다운로드하려면 [public API]({{< relref path="/ref/python/public-api/api.md" lang="ko" >}})를 사용하세요:

```python
import wandb

api = wandb.Api()
run = api.run('<entity>/<project>/<run_id>')
run.file('<file>').download()

for f in run.files():
    if <condition>:
        f.download()
```
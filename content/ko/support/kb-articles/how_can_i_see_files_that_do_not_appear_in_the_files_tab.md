---
title: How can I see files that do not appear in the Files tab?
menu:
  support:
    identifier: ko-support-kb-articles-how_can_i_see_files_that_do_not_appear_in_the_files_tab
support:
- experiments
toc_hide: true
type: docs
url: /ko/support/:filename
---

파일 탭은 최대 10,000개의 파일을 보여줍니다. 모든 파일을 다운로드하려면 [public API]({{< relref path="/ref/python/public-api/api.md" lang="ko" >}})를 사용하세요.

```python
import wandb

api = wandb.Api()
run = api.run('<entity>/<project>/<run_id>')
run.file('<file>').download()

for f in run.files():
    if <condition>:
        f.download()
```

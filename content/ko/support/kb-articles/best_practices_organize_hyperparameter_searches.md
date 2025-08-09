---
title: 하이퍼파라미터 탐색을 체계적으로 관리하는 모범 사례
menu:
  support:
    identifier: ko-support-kb-articles-best_practices_organize_hyperparameter_searches
support:
- 하이퍼파라미터
- Sweeps
- run
toc_hide: true
type: docs
url: /support/:filename
---

고유한 태그는 `wandb.init(tags='your_tag')`로 설정할 수 있습니다. 이렇게 하면 Project Page의 Runs Table에서 해당 태그로 Projects 의 Run 을 효율적으로 필터링할 수 있습니다.

`wandb.init()`에 대한 자세한 내용은 [`wandb.init()` 참고 문서]({{< relref path="/ref/python/sdk/functions/init.md" lang="ko" >}})를 확인하세요.
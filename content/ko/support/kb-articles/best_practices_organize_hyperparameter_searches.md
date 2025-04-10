---
title: Best practices to organize hyperparameter searches
menu:
  support:
    identifier: ko-support-kb-articles-best_practices_organize_hyperparameter_searches
support:
- hyperparameter
- sweeps
- runs
toc_hide: true
type: docs
url: /support/:filename
---

`wandb.init(tags='your_tag')` 로 고유 태그를 설정합니다. 이렇게 하면 프로젝트 페이지의 Runs Table에서 해당 태그를 선택하여 프로젝트 run을 효율적으로 필터링할 수 있습니다.

wandb.int에 대한 자세한 내용은 [documentation]({{< relref path="/ref/python/init.md" lang="ko" >}})을 참조하세요.

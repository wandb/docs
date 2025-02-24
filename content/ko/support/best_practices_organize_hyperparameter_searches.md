---
title: Best practices to organize hyperparameter searches
menu:
  support:
    identifier: ko-support-best_practices_organize_hyperparameter_searches
tags:
- hyperparameter
- sweeps
- runs
toc_hide: true
type: docs
---

`wandb.init(tags='your_tag')` 로 고유한 태그를 설정하세요. 이렇게 하면 프로젝트 페이지의 Runs Table에서 해당 태그를 선택하여 프로젝트 run을 효율적으로 필터링할 수 있습니다.

wandb.int에 대한 자세한 내용은 [설명서]({{< relref path="/ref/python/init.md" lang="ko" >}})를 참조하세요.

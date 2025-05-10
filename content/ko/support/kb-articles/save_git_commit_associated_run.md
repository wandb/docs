---
title: How can I save the git commit associated with my run?
menu:
  support:
    identifier: ko-support-kb-articles-save_git_commit_associated_run
support:
- experiments
toc_hide: true
type: docs
url: /ko/support/:filename
translationKey: save_git_commit_associated_run
---
`wandb.init` 가 호출되면 시스템은 원격 저장소 링크와 최신 커밋의 SHA를 포함한 git 정보를 자동으로 수집합니다. 이 정보는 [run 페이지]({{< relref path="/guides/models/track/runs/#view-logged-runs" lang="ko" >}})에 나타납니다. 이 정보를 보려면 스크립트를 실행할 때 현재 작업 디렉토리가 git으로 관리되는 폴더 내에 있는지 확인하세요.

git 커밋과 실험을 실행하는 데 사용된 코맨드는 사용자에게는 계속 표시되지만 외부 사용자에게는 숨겨집니다. 공개 Projects에서는 이러한 세부 정보가 비공개로 유지됩니다.

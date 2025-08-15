---
title: 내 run 에 연관된 git 커밋을 어떻게 저장할 수 있나요?
menu:
  support:
    identifier: ko-support-kb-articles-save_git_commit_associated_run
support:
- 실험
toc_hide: true
type: docs
url: /support/:filename
---

`wandb.init`이 호출될 때, 시스템은 원격 저장소 링크와 최신 커밋의 SHA 를 포함한 git 정보를 자동으로 수집합니다. 이 정보는 [run 페이지]({{< relref path="/guides/models/track/runs/#view-logged-runs" lang="ko" >}})에서 확인할 수 있습니다. 해당 정보를 보려면, 스크립트를 실행할 때 현재 작업 디렉토리가 git 으로 관리되는 폴더 내에 있어야 합니다.

git 커밋과 실험을 실행한 코맨드는 사용자에게는 계속 보이지만, 외부 사용자에게는 숨겨집니다. 퍼블릭 프로젝트의 경우에도 이러한 정보는 비공개로 유지됩니다.
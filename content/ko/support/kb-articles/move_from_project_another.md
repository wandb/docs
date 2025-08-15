---
title: 한 프로젝트에서 다른 프로젝트로 run 을 이동할 수 있나요?
menu:
  support:
    identifier: ko-support-kb-articles-move_from_project_another
support:
- run
toc_hide: true
type: docs
url: /support/:filename
---

run 을(를) 한 프로젝트에서 다른 프로젝트로 옮기려면 다음 단계를 따르세요.

- 옮길 run 이 포함된 프로젝트 페이지로 이동합니다.
- **Runs** 탭을 클릭하여 runs 테이블을 엽니다.
- 옮길 run 들을 선택합니다.
- **Move** 버튼을 클릭합니다.
- 이동할 대상 프로젝트를 선택하고 작업을 확인합니다.

W&B 는 UI 를 통해 run 이동만 지원하며, run 복사는 지원하지 않습니다. run 과 함께 로그된 Artifacts 는 새 프로젝트로 자동으로 옮겨지지 않습니다. Artifacts 를 새 위치로 수동으로 옮기려면 [`wandb artifact get`]({{< relref path="/ref/cli/wandb-artifact/wandb-artifact-get/" lang="ko" >}}) SDK 코맨드나 [`Api.artifact` API]({{< relref path="/ref/python/public-api/api/#artifact" lang="ko" >}}) 를 사용해 artifact 를 다운로드한 뒤, [wandb artifact put]({{< relref path="/ref/cli/wandb-artifact/wandb-artifact-put/" lang="ko" >}}) 또는 `Api.artifact` API 로 run 의 새 위치에 업로드할 수 있습니다.
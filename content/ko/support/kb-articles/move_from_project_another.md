---
title: Is it possible to move a run from one project to another?
menu:
  support:
    identifier: ko-support-kb-articles-move_from_project_another
support:
- runs
toc_hide: true
type: docs
url: /support/:filename
---

다음 단계를 따라 한 프로젝트에서 다른 프로젝트로 run을 이동할 수 있습니다.

- 이동할 run이 있는 프로젝트 페이지로 이동합니다.
- **Runs** 탭을 클릭하여 run 테이블을 엽니다.
- 이동할 run을 선택합니다.
- **Move** 버튼을 클릭합니다.
- 대상 프로젝트를 선택하고 작업을 확인합니다.

W&B는 UI를 통한 run 이동을 지원하지만, run 복사는 지원하지 않습니다. run과 함께 로그된 Artifacts는 새 프로젝트로 전송되지 않습니다. Artifacts를 run의 새 위치로 수동으로 이동하려면 [`wandb artifact get`]({{< relref path="/ref/cli/wandb-artifact/wandb-artifact-get/" lang="ko" >}}) SDK 코맨드 또는 [`Api.artifact` API]({{< relref path="/ref/python/public-api/api/#artifact" lang="ko" >}})를 사용하여 아티팩트를 다운로드한 다음 [wandb artifact put]({{< relref path="/ref/cli/wandb-artifact/wandb-artifact-put/" lang="ko" >}}) 또는 `Api.artifact` API를 사용하여 run의 새 위치로 업로드할 수 있습니다.

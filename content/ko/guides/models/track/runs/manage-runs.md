---
title: run 이동하기
menu:
  default:
    identifier: ko-guides-models-track-runs-manage-runs
    parent: what-are-runs
---

이 페이지에서는 run 을 한 프로젝트에서 다른 프로젝트로, 혹은 팀 안팎으로 또는 한 팀에서 다른 팀으로 이동하는 방법을 안내합니다. 현재 위치와 이동할 위치 모두에 대해 run 에 접근할 수 있어야 합니다.

{{% alert %}}
run 을 이동해도, 연관된 기존 Artifacts 는 함께 이동되지 않습니다. artifact 를 수동으로 이동하려면 [`wandb artifact get`]({{< relref path="/ref/cli/wandb-artifact/wandb-artifact-get/" lang="ko" >}}) SDK 명령어나 [`Api.artifact` API]({{< relref path="/ref/python/public-api/api/#artifact" lang="ko" >}}) 를 사용해 artifact 를 다운로드한 후, [`wandb artifact put`]({{< relref path="/ref/cli/wandb-artifact/wandb-artifact-put/" lang="ko" >}}) 또는 `Api.artifact` API 를 사용하여 해당 run 이 옮겨진 위치에 artifact 를 업로드할 수 있습니다.
{{% /alert %}}

**Runs** 탭을 사용자 정의하는 방법은 [Project page]({{< relref path="/guides/models/track/project-page.md#runs-tab" lang="ko" >}}) 문서를 참고하세요.

run 을 experiment 기준으로 그룹화하려면 [Set a group in the UI]({{< relref path="grouping.md#set-a-group-in-the-ui" lang="ko" >}}) 가이드를 참고하세요.

## 프로젝트 간에 run 이동하기

한 프로젝트에서 다른 프로젝트로 run 을 이동하려면:

1. 이동하려는 run 이 포함된 프로젝트로 이동합니다.
2. 프로젝트 사이드바에서 **Runs** 탭을 클릭합니다.
3. 이동할 run 옆 체크박스를 선택합니다.
4. 테이블 상단의 **Move** 버튼을 클릭합니다.
5. 드롭다운에서 목적지 프로젝트를 선택합니다.

{{< img src="/images/app_ui/howto_move_runs.gif" alt="프로젝트 간 run 이동 데모" >}}

## 팀으로 run 이동하기

자신이 소속된 팀으로 run 을 이동하려면:

1. 이동하려는 run 이 포함된 프로젝트로 이동합니다.
2. 프로젝트 사이드바에서 **Runs** 탭을 클릭합니다.
3. 이동할 run 옆 체크박스를 선택합니다.
4. 테이블 상단의 **Move** 버튼을 클릭합니다.
5. 드롭다운에서 목적지 팀과 프로젝트를 선택합니다.

{{< img src="/images/app_ui/demo_move_runs.gif" alt="팀으로 run 이동 데모" >}}
---
title: 내 run 의 상태가 UI에서는 `crashed`로 표시되는데, 내 컴퓨터에서는 아직 실행 중입니다. 내 데이터를 복구하려면 어떻게 해야
  하나요?
menu:
  support:
    identifier: ko-support-kb-articles-runs_state_crashed_ui_running_machine_get_data
support:
- 실험
toc_hide: true
type: docs
url: /support/:filename
---

트레이닝 중에 머신과의 연결이 끊겼을 가능성이 높습니다. [`wandb sync [PATH_TO_RUN]`]({{< relref path="/ref/cli/wandb-sync.md" lang="ko" >}}) 명령어를 실행하여 데이터를 복구할 수 있습니다. run 경로는 현재 진행 중인 run 의 Run ID 와 일치하는 `wandb` 디렉토리 내의 폴더입니다.
---
title: My run's state is `crashed` on the UI but is still running on my machine. What
  do I do to get my data back?
menu:
  support:
    identifier: ko-support-kb-articles-runs_state_crashed_ui_running_machine_get_data
support:
- experiments
toc_hide: true
type: docs
url: /ko/support/:filename
translationKey: runs_state_crashed_ui_running_machine_get_data
---
트레이닝 도중 머신 연결이 끊겼을 가능성이 있습니다. [`wandb sync [PATH_TO_RUN]`]({{< relref path="/ref/cli/wandb-sync.md" lang="ko" >}}) 을 실행하여 데이터를 복구하세요. 해당 run의 경로는 실행 중인 Run ID와 일치하는 `wandb` 디렉토리의 폴더입니다.

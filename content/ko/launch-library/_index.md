---
title: 'launch-library

  '
cascade:
  hidden: true
  menu:
    launch-library:
      parent: launch-library
  type: docs
hidden: true
menu:
  launch:
    identifier: launch-library
type: docs
---

## 클래스

[`class LaunchAgent`](./launchagent.md): Launch 에이전트 클래스이며, 지정된 run 큐를 폴링하여 wandb launch를 위한 run을 실행합니다.

## 함수

[`launch(...)`](./launch.md): W&B launch experiment를 실행합니다.

[`launch_add(...)`](./launch_add.md): W&B launch experiment를 큐에 추가합니다. source uri, job, 또는 docker_image 중 하나를 이용할 수 있습니다.
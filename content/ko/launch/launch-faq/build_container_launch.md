---
title: W&B가 컨테이너를 만들어주지 않길 원하는데, 그래도 Launch를 사용할 수 있나요?
menu:
  launch:
    identifier: ko-launch-launch-faq-build_container_launch
    parent: launch-faq
---

사전 구축된 Docker 이미지를 실행하려면, 아래 코맨드를 입력하세요. `<>` 안에는 여러분의 정보를 입력해야 합니다.

```bash
wandb launch -d <docker-image-uri> -q <queue-name> -E <entrypoint>
```

이 코맨드는 job을 생성하고 run을 시작합니다.

이미지로부터 job을 생성하려면, 아래 코맨드를 사용하세요.

```bash
wandb job create image <image-name> -p <project> -e <entity>
```
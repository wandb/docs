---
title: I do not want W&B to build a container for me, can I still use Launch?
menu:
  launch:
    identifier: ko-launch-launch-faq-build_container_launch
    parent: launch-faq
---

미리 빌드된 Docker 이미지를 시작하려면 다음 코맨드를 실행하세요. `<>`의 자리 표시자를 특정 정보로 바꾸세요.

```bash
wandb launch -d <docker-image-uri> -q <queue-name> -E <entrypoint>
```

이 코맨드는 job을 생성하고 run을 시작합니다.

이미지에서 job을 생성하려면 다음 코맨드를 사용하세요.

```bash
wandb job create image <image-name> -p <project> -e <entity>
```

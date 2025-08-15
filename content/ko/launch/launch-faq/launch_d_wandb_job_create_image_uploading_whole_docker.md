---
title: '`wandb launch -d` 또는 `wandb job create image` 명령어가 도커 아티팩트 전체를 업로드하고 레지스트리에서
  가져오지 않는 건가요?'
menu:
  launch:
    identifier: ko-launch-launch-faq-launch_d_wandb_job_create_image_uploading_whole_docker
    parent: launch-faq
---

아니요, `wandb launch -d` 코맨드는 이미지를 레지스트리에 업로드하지 않습니다. 이미지는 별도로 레지스트리에 업로드해야 합니다. 다음 단계를 따라 진행하세요.

1. 이미지를 빌드합니다.
2. 이미지를 레지스트리에 푸시합니다.

워크플로우는 다음과 같습니다:

```bash
docker build -t <repo-url>:<tag> .
docker push <repo-url>:<tag>
wandb launch -d <repo-url>:<tag>
```

Launch 에이전트는 지정한 컨테이너를 가리키는 작업을 실행합니다. 컨테이너 레지스트리에서 이미지를 가져오기 위한 에이전트 엑세스 설정 예시는 [고급 에이전트 설정]({{< relref path="/launch/set-up-launch/setup-agent-advanced.md#agent-configuration" lang="ko" >}})을 참고하세요.

Kubernetes 의 경우, Kubernetes 클러스터의 파드가 이미지가 푸시된 레지스트리에 엑세스할 수 있는지 확인하세요.
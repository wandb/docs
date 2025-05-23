---
title: Is `wandb launch -d` or `wandb job create image` uploading a whole docker artifact
  and not pulling from a registry?
menu:
  launch:
    identifier: ko-launch-launch-faq-launch_d_wandb_job_create_image_uploading_whole_docker
    parent: launch-faq
---

아니요, `wandb launch -d` 코맨드는 이미지를 레지스트리에 업로드하지 않습니다. 이미지를 레지스트리에 별도로 업로드하세요. 다음 단계를 따르세요.

1. 이미지를 빌드합니다.
2. 이미지를 레지스트리에 푸시합니다.

워크플로우는 다음과 같습니다.

```bash
docker build -t <repo-url>:<tag> .
docker push <repo-url>:<tag>
wandb launch -d <repo-url>:<tag>
```

그러면 Launch 에이전트가 지정된 컨테이너를 가리키는 작업을 시작합니다. 컨테이너 레지스트리에서 이미지를 가져오기 위해 에이전트 엑세스를 구성하는 방법에 대한 예는 [고급 에이전트 설정]({{< relref path="/launch/set-up-launch/setup-agent-advanced.md#agent-configuration" lang="ko" >}})을 참조하세요.

Kubernetes의 경우 Kubernetes 클러스터 포드가 이미지가 푸시되는 레지스트리에 엑세스할 수 있는지 확인하세요.

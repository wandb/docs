---
title: Dockerfile 을 지정해서 W&B 가 Docker 이미지를 대신 빌드하게 할 수 있나요?
menu:
  launch:
    identifier: ko-launch-launch-faq-dockerfile_let_wb_build_docker_image_me
    parent: launch-faq
---

이 기능은 요구 사항은 안정적이지만 코드베이스가 자주 변경되는 프로젝트에 적합합니다.

{{% alert color="secondary" %}}
Dockerfile 을 마운트 방식을 사용하도록 포매팅하세요. 자세한 내용은 [Docker Docs 웹사이트의 Mounts 문서](https://docs.docker.com/build/guide/mounts/)를 참고하세요.
{{% /alert %}}

Dockerfile 을 구성한 후에는 다음 세 가지 방법 중 하나로 W&B 에 지정할 수 있습니다:

* Dockerfile.wandb 사용
* W&B CLI 사용
* W&B App 사용

{{< tabpane text=true >}}
{{% tab "Dockerfile.wandb" %}}
W&B run 의 엔트리포인트와 같은 디렉토리에 `Dockerfile.wandb` 파일을 포함하세요. W&B 는 이 파일을 내장 Dockerfile 대신 사용합니다.
{{% /tab %}}
{{% tab "W&B CLI" %}}
`wandb launch` 코맨드에서 `--dockerfile` 플래그를 사용해 job 을 큐에 올릴 수 있습니다:

```bash
wandb launch --dockerfile path/to/Dockerfile
```
{{% /tab %}}
{{% tab "W&B app" %}}
W&B App 에서 job 을 큐에 추가할 때, **Overrides** 섹션에 Dockerfile 경로를 입력하세요. `"dockerfile"` 을 키로, Dockerfile 경로를 값으로 하는 키-값 쌍으로 입력해야 합니다.

아래 JSON 예시는 로컬 디렉토리에서 Dockerfile 을 포함하는 방법을 보여줍니다:

```json title="Launch job W&B App"
{
  "args": [],
  "run_config": {
    "lr": 0,
    "batch_size": 0,
    "epochs": 0
  },
  "entrypoint": [],
  "dockerfile": "./Dockerfile"
}
```
{{% /tab %}}
{{% /tabpane %}}
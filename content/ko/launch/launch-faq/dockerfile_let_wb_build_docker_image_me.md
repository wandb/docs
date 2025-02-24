---
title: Can I specify a Dockerfile and let W&B build a Docker image for me?
menu:
  launch:
    identifier: ko-launch-launch-faq-dockerfile_let_wb_build_docker_image_me
    parent: launch-faq
---

이 기능은 안정적인 요구 사항을 가지고 있지만 코드 베이스가 자주 변경되는 프로젝트에 적합합니다.

{{% alert color="secondary" %}}
마운트를 사용하도록 Dockerfile의 형식을 지정하세요. 자세한 내용은 [Docker Docs 웹사이트의 마운트 관련 문서](https://docs.docker.com/build/guide/mounts/)를 참조하세요.
{{% /alert %}}

Dockerfile을 구성한 후에는 다음 세 가지 방법 중 하나로 W&B에 지정합니다.

* Dockerfile.wandb 사용
* W&B CLI 사용
* W&B App 사용

{{< tabpane text=true >}}
{{% tab "Dockerfile.wandb" %}}
W&B run의 진입점과 동일한 디렉토리에 `Dockerfile.wandb` 파일을 포함합니다. W&B는 내장 Dockerfile 대신 이 파일을 사용합니다.
{{% /tab %}}
{{% tab "W&B CLI" %}}
`wandb launch` 코맨드와 함께 `--dockerfile` 플래그를 사용하여 작업을 대기열에 추가합니다.

```bash
wandb launch --dockerfile path/to/Dockerfile
```
{{% /tab %}}
{{% tab "W&B app" %}}
W&B App에서 작업을 대기열에 추가할 때 **Overrides** 섹션에서 Dockerfile 경로를 제공합니다. `"dockerfile"`을 키로, Dockerfile의 경로를 값으로 하여 키-값 쌍으로 입력합니다.

다음 JSON은 로컬 디렉토리에 Dockerfile을 포함하는 방법을 보여줍니다.

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

---
title: '튜토리얼: W&B Launch 기본 사용법'
description: W&B Launch 시작 가이드.
menu:
  launch:
    identifier: ko-launch-walkthrough
    parent: launch
url: guides/launch/walkthrough
weight: 1
---

## Launch란 무엇인가요? 

{{< cta-button colabLink="https://colab.research.google.com/drive/1wX0OSVxZJDHRsZaOaOEDx-lLUrO1hHgP" >}}

W&B Launch를 사용하면 데스크톱에서 Amazon SageMaker, Kubernetes 등과 같은 컴퓨팅 리소스로의 트레이닝 [runs]({{< relref path="/guides/models/track/runs/" lang="ko" >}}) 확장이 아주 쉽습니다. W&B Launch가 설정되면, 몇 번의 클릭과 코맨드로 트레이닝 스크립트 실행, 모델 평가, 프로덕션 추론용 모델 준비 등 다양한 작업을 빠르게 할 수 있습니다. 

## 작동 방식

Launch는 세 가지 핵심 구성 요소로 이루어져 있습니다: **launch jobs**, **queues**, 그리고 **agents**.

[*launch job*]({{< relref path="./launch-terminology.md#launch-job" lang="ko" >}})은 ML 워크플로우 내에서 작업을 구성하고 실행하는 청사진 역할을 합니다. launch job이 준비되면, 이를 [*launch queue*]({{< relref path="./launch-terminology.md#launch-queue" lang="ko" >}})에 추가할 수 있습니다. launch queue는 선입선출(FIFO) 방식의 큐로, Amazon SageMaker나 Kubernetes 클러스터 같은 특정 컴퓨트 타겟 리소스에 작업을 제출하고 설정할 수 있습니다. 

작업이 큐에 추가되면, [*launch agent*]({{< relref path="./launch-terminology.md#launch-agent" lang="ko" >}})가 해당 큐를 폴링하여, 큐가 지정한 시스템에서 작업을 실행합니다.

{{< img src="/images/launch/launch_overview.png" alt="W&B Launch overview diagram" >}}

유스 케이스에 따라 여러분 또는 팀원이 [compute resource target]({{< relref path="./launch-terminology.md#target-resources" lang="ko" >}})(예: Amazon SageMaker)에 맞게 launch queue를 설정하고, 자체 인프라에 launch agent를 배포하게 됩니다.

Launch에 대한 더 자세한 용어와 개념은 [Terms and concepts]({{< relref path="./launch-terminology.md" lang="ko" >}}) 페이지를 참고하세요.

## 시작 방법

유스 케이스에 따라 W&B Launch를 시작하려면 아래 리소스를 참고하세요:

* W&B Launch를 처음 사용하신다면, [Launch walkthrough]({{< relref path="#walkthrough" lang="ko" >}}) 가이드를 따라해보는 것을 추천합니다.
* [W&B Launch 설정 방법]({{< relref path="/launch/set-up-launch/" lang="ko" >}})을 알아보세요.
* [launch job 만들기]({{< relref path="./create-and-deploy-jobs/create-launch-job.md" lang="ko" >}}).
* [Deploying to Triton](https://github.com/wandb/launch-jobs/tree/main/jobs/deploy_to_nvidia_triton), [evaluating an LLM](https://github.com/wandb/launch-jobs/tree/main/jobs/openai_evals) 등 다양한 공통 작업 템플릿을 [public jobs GitHub repository](https://github.com/wandb/launch-jobs)에서 확인할 수 있습니다.
    * 해당 저장소에서 생성된 launch job은 공개 [`wandb/jobs` project](https://wandb.ai/wandb/jobs/jobs) W&B 프로젝트에서 확인할 수 있습니다.

## Walkthrough

이 페이지에서는 W&B Launch 워크플로우의 기본 과정을 안내합니다.

{{% alert %}}
W&B Launch는 기계학습 워크로드를 컨테이너에서 실행합니다. 컨테이너에 익숙하지 않아도 무방하지만, 이해하시면 이 워크스루에 도움이 됩니다. 컨테이너에 대해 간단히 알고 싶다면 [Docker 공식 문서](https://docs.docker.com/guides/docker-concepts/the-basics/what-is-a-container/)를 참고하세요.
{{% /alert %}}

## 사전 준비 사항

시작하기 전에 다음 사전 조건을 충족했는지 확인하세요:

1. https://wandb.ai/site에서 계정을 생성한 후, W&B 계정에 로그인하세요.
2. 이 walkthrough에는 Docker CLI 및 엔진이 설치된 머신에 터미널 엑세스가 필요합니다. 자세한 내용은 [Docker 설치 가이드](https://docs.docker.com/engine/install/)를 참고하세요.
3. W&B Python SDK 버전 `0.17.1` 이상을 설치하세요:
```bash
pip install wandb>=0.17.1
```
4. 터미널에서 `wandb login`을 실행하거나 `WANDB_API_KEY` 환경 변수를 설정하여 W&B 인증을 진행하세요.

{{< tabpane text=true >}}
{{% tab "W&B 로그인" %}}
    터미널에서 아래 명령어를 실행하세요:
    
    ```bash
    wandb login
    ```
{{% /tab %}}
{{% tab "환경 변수" %}}

    ```bash
    WANDB_API_KEY=<your-api-key>
    ```

    `<your-api-key>`에 본인의 W&B API 키를 입력하세요.
{{% /tab %}}
{{% /tabpane %}}

## launch job 생성하기
[launch job]({{< relref path="./launch-terminology.md#launch-job" lang="ko" >}}) 생성은 세 가지 방법 중 하나로 할 수 있습니다: 도커 이미지 사용, 깃 리포지터리에서, 또는 로컬 소스 코드에서:

{{< tabpane text=true >}}
{{% tab "Docker 이미지로" %}}
미리 만들어진 컨테이너를 실행해 W&B에 메시지를 로그하려면 터미널을 열고 아래 명령을 실행하세요.

```bash
wandb launch --docker-image wandb/job_hello_world:main --project launch-quickstart
```

위 명령은 `wandb/job_hello_world:main` 컨테이너 이미지를 다운로드하고 실행합니다.

Launch가 컨테이너를 설정해 `wandb`로 기록된 모든 로그를 `launch-quickstart` 프로젝트로 보냅니다. 컨테이너가 W&B로 메시지를 로그하고, 새로 생성된 run으로 연결되는 링크가 표시됩니다. 링크를 클릭하면 W&B UI에서 run을 확인할 수 있습니다.
{{% /tab %}}
{{% tab "Git 리포지터리에서" %}}
[W&B Launch jobs repository의 소스 코드](https://github.com/wandb/launch-jobs)에서 같은 헬로월드 job을 실행하려면:

```bash
wandb launch --uri https://github.com/wandb/launch-jobs.git \\
--job-name hello-world-git --project launch-quickstart \\ 
--build-context jobs/hello_world --dockerfile Dockerfile.wandb \\ 
--entry-point "python job.py"
```
이 명령어는 다음을 수행합니다:
1. [W&B Launch jobs 리포지터리](https://github.com/wandb/launch-jobs)를 임시 디렉토리에 클론합니다.
2. **hello** 프로젝트에 **hello-world-git**이라는 job을 생성합니다. 이 job은 정확히 어떤 소스 코드·설정이 실행됐는지 추적합니다.
3. `jobs/hello_world` 디렉토리와 `Dockerfile.wandb`로부터 컨테이너 이미지를 빌드합니다.
4. 컨테이너를 시작하고 `job.py` 파이썬 스크립트를 실행합니다.

출력 결과로 이미지 빌드 및 실행 로그가 나타나며, 컨테이너의 동작은 앞선 예와 거의 동일합니다.

{{% /tab %}}
{{% tab "로컬 소스 코드에서" %}}

Git 리포지터리에 버전 관리되어 있지 않은 코드도 `--uri` 인수에 로컬 디렉토리 경로를 지정해 launch할 수 있습니다.

빈 디렉토리를 하나 만들고, 아래 내용을 담은 `train.py` 파이썬 스크립트를 추가하세요:

```python
import wandb

with wandb.init() as run:
    run.log({"hello": "world"})
```

그리고 아래 내용을 가진 `requirements.txt` 파일을 추가하세요:

```text
wandb>=0.17.1
```

이 디렉토리에서 다음 명령어를 실행하세요:

```bash
wandb launch --uri . --job-name hello-world-code --project launch-quickstart --entry-point "python train.py"
```

이 명령어는 다음을 수행합니다:
1. 현재 디렉토리의 내용을 W&B에 Code Artifact로 로그합니다.
2. **launch-quickstart** 프로젝트에 **hello-world-code** 라는 job을 만듭니다.
3. `train.py`와 `requirements.txt`를 기본 이미지에 복사하고, 필요 패키지를 `pip install`하여 컨테이너 이미지를 빌드합니다.
4. 컨테이너를 시작하고 `python train.py`를 실행합니다.
{{% /tab %}}
{{< /tabpane >}}

## 큐 생성하기

Launch는 팀이 공유 컴퓨트 환경에서 워크플로우를 만들도록 돕기 위해 설계되었습니다. 지금까지의 예시는 `wandb launch` 명령이 로컬 머신에서 컨테이너를 동기적으로 실행했습니다. Launch queues와 agents는 공유 리소스에서의 비동기 실행과 우선순위 지정, 하이퍼파라미터 최적화 등 고급 기능을 지원합니다. 기본 queue를 만들려면 아래 단계를 따라하세요:

1. [wandb.ai/launch](https://wandb.ai/launch)로 이동 후 **Create a queue** 버튼을 클릭하세요.
2. 큐를 연동할 **Entity**를 선택하세요.
3. **Queue name**을 입력하세요.
4. **Resource**로 **Docker**를 선택하세요.
5. **Configuration**은 잠시 비워두세요.
6. **Create queue**를 클릭하세요 :rocket:

버튼을 클릭하면 브라우저가 큐 뷰의 **Agents** 탭으로 이동합니다. 큐는 에이전트가 시작될 때까지 **Not active** 상태로 남아있습니다.

{{< img src="/images/launch/create_docker_queue.gif" alt="Docker queue creation" >}}

고급 queue 설정 방법은 [advanced queue setup page]({{< relref path="/launch/set-up-launch/setup-queue-advanced.md" lang="ko" >}})를 참고하세요.

## 큐에 agent 연결하기

에이전트가 폴링하고 있지 않으면 큐 뷰 상단에 빨간 배너로 **Add an agent** 버튼이 표시됩니다. 버튼을 클릭하면 agent를 실행할 명령어를 복사할 수 있습니다. 예시는 아래와 같습니다:

```bash
wandb launch-agent --queue <queue-name> --entity <entity-name>
```

이 명령을 터미널에 입력하면 에이전트가 시작되어, 지정한 queue에서 실행할 job을 폴링합니다. job을 받으면, agent가 컨테이너 이미지를 다운로드/빌드 후 로컬에서 `wandb launch`를 실행하는 것처럼 job을 수행합니다.

다시 [Launch 페이지](https://wandb.ai/launch)로 돌아가 큐가 **Active** 상태가 되었는지 확인하세요.

## 큐에 작업 제출하기

본인 W&B 계정에서 **launch-quickstart** 프로젝트로 이동한 뒤 좌측 네비게이션에서 jobs 탭을 엽니다.

**Jobs** 페이지에서는 이전에 실행된 run에서 생성된 W&B Jobs 목록이 표시됩니다. 본인의 launch job을 클릭해, 소스 코드·의존성·job에서 생성된 run 등을 확인할 수 있습니다. walkthrough까지 마치면 목록에 3개의 job이 있어야 합니다.

생성된 job 중 하나를 선택해서 아래와 같이 큐로 제출해보세요:

1. **Launch** 버튼을 눌러 job을 큐로 제출하세요. **Launch** 드로어가 나타납니다.
2. 이전에 만든 **Queue**를 선택한 후 **Launch**를 클릭하세요.

이렇게 하면 job이 큐에 제출됩니다. 이 큐를 폴링 중인 agent가 job을 받아 실행합니다. job의 진행 상황은 W&B UI 또는 터미널의 agent 출력을 통해 확인할 수 있습니다.

`wandb launch` 명령에 `--queue` 인수를 사용하면 job을 직접 큐에 제출할 수도 있습니다. 예를 들어, hello-world 컨테이너 job을 큐로 제출하려면 아래와 같이 실행하세요:

```bash
wandb launch --docker-image wandb/job_hello_world:main --project launch-quickstart --queue <queue-name>
```
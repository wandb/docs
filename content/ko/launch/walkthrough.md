---
title: 'Tutorial: W&B Launch basics'
description: W&B Launch 시작하기 가이드.
menu:
  launch:
    identifier: ko-launch-walkthrough
    parent: launch
url: guides/launch/walkthrough
weight: 1
---

## Launch 란 무엇인가?

{{< cta-button colabLink="https://colab.research.google.com/drive/1wX0OSVxZJDHRsZaOaOEDx-lLUrO1hHgP" >}}

W&B Launch를 사용하면 데스크톱에서 Amazon SageMaker, Kubernetes 등과 같은 컴퓨팅 리소스로 트레이닝 [runs]({{< relref path="/guides/models/track/runs/" lang="ko" >}})을 쉽게 확장할 수 있습니다. W&B Launch가 구성되면 몇 번의 클릭과 코맨드로 트레이닝 스크립트, 모델 평가 스위트 실행, 프로덕션 추론을 위한 모델 준비 등을 빠르게 수행할 수 있습니다.

## 작동 방식

Launch는 **launch jobs**, **queues**, **agents**의 세 가지 기본 구성 요소로 구성됩니다.

[*launch job*]({{< relref path="./launch-terminology.md#launch-job" lang="ko" >}})은 ML 워크플로우에서 작업을 구성하고 실행하기 위한 청사진입니다. launch job이 있으면 [*launch queue*]({{< relref path="./launch-terminology.md#launch-queue" lang="ko" >}})에 추가할 수 있습니다. launch queue는 선입선출(FIFO) 큐로, Amazon SageMaker 또는 Kubernetes 클러스터와 같은 특정 컴퓨팅 대상 리소스에 작업을 구성하고 제출할 수 있습니다.

작업이 큐에 추가되면 [*launch agents*]({{< relref path="./launch-terminology.md#launch-agent" lang="ko" >}})는 해당 큐를 폴링하고 큐가 대상으로 하는 시스템에서 작업을 실행합니다.

{{< img src="/images/launch/launch_overview.png" alt="" >}}

유스 케이스에 따라 귀하(또는 팀의 누군가)는 선택한 [컴퓨팅 리소스 대상]({{< relref path="./launch-terminology.md#target-resources" lang="ko" >}}) (예: Amazon SageMaker)에 따라 launch queue를 구성하고 자체 인프라에 launch agent를 배포합니다.

launch jobs, 큐 작동 방식, launch agents 및 W&B Launch 작동 방식에 대한 추가 정보는 [용어 및 개념]({{< relref path="./launch-terminology.md" lang="ko" >}}) 페이지를 참조하세요.

## 시작 방법

유스 케이스에 따라 다음 리소스를 탐색하여 W&B Launch를 시작하세요.

* W&B Launch를 처음 사용하는 경우 [워크스루]({{< relref path="#walkthrough" lang="ko" >}}) 가이드를 살펴보는 것이 좋습니다.
* [W&B Launch 설정 방법]({{< relref path="/launch/set-up-launch/" lang="ko" >}})을 알아봅니다.
* [launch job 만들기]({{< relref path="./create-and-deploy-jobs/create-launch-job.md" lang="ko" >}}).
* W&B Launch [공개 jobs GitHub 저장소](https://github.com/wandb/launch-jobs)에서 [Triton에 배포](https://github.com/wandb/launch-jobs/tree/main/jobs/deploy_to_nvidia_triton), [LLM 평가](https://github.com/wandb/launch-jobs/tree/main/jobs/openai_evals) 등과 같은 일반적인 작업 템플릿을 확인하세요.
    * 이 저장소에서 생성된 launch jobs은 이 공개 [`wandb/jobs` project](https://wandb.ai/wandb/jobs/jobs) W&B 프로젝트에서 볼 수 있습니다.

## 워크스루

이 페이지에서는 W&B Launch 워크플로우의 기본 사항을 살펴봅니다.

{{% alert %}}
W&B Launch는 컨테이너에서 기계 학습 워크로드를 실행합니다. 컨테이너에 대한 지식이 필수는 아니지만 이 워크스루에 도움이 될 수 있습니다. 컨테이너에 대한 입문서는 [Docker 설명서](https://docs.docker.com/guides/docker-concepts/the-basics/what-is-a-container/)를 참조하세요.
{{% /alert %}}

## 전제 조건

시작하기 전에 다음 전제 조건을 충족했는지 확인하세요.

1. https://wandb.ai/site에서 계정에 가입한 다음 W&B 계정에 로그인합니다.
2. 이 워크스루에서는 작동하는 Docker CLI 및 엔진이 있는 시스템에 대한 터미널 엑세스가 필요합니다. 자세한 내용은 [Docker 설치 가이드](https://docs.docker.com/engine/install/)를 참조하세요.
3. W&B Python SDK 버전 `0.17.1` 이상을 설치합니다.
```bash
pip install wandb>=0.17.1
```
4. 터미널 내에서 `wandb login`을 실행하거나 `WANDB_API_KEY` 환경 변수를 설정하여 W&B로 인증합니다.

{{< tabpane text=true >}}
{{% tab "W&B에 로그인" %}}
    터미널 내에서 다음을 실행합니다.
    
    ```bash
    wandb login
    ```
{{% /tab %}}
{{% tab "환경 변수" %}}

    ```bash
    WANDB_API_KEY=<your-api-key>
    ```

    `<your-api-key>`를 W&B API 키로 바꿉니다.
{{% /tab %}}
{{< /tabpane >}}

## launch job 만들기
Docker 이미지, Git 저장소 또는 로컬 소스 코드를 사용하여 세 가지 방법 중 하나로 [launch job]({{< relref path="./launch-terminology.md#launch-job" lang="ko" >}})을 만듭니다.

{{< tabpane text=true >}}
{{% tab "Docker 이미지 사용" %}}
W&B에 메시지를 기록하는 미리 만들어진 컨테이너를 실행하려면 터미널을 열고 다음 코맨드를 실행합니다.

```bash
wandb launch --docker-image wandb/job_hello_world:main --project launch-quickstart
```

이전 코맨드는 컨테이너 이미지 `wandb/job_hello_world:main`을 다운로드하여 실행합니다.

Launch는 `wandb`로 기록된 모든 항목을 `launch-quickstart` 프로젝트에 보고하도록 컨테이너를 구성합니다. 컨테이너는 W&B에 메시지를 기록하고 새로 생성된 run에 대한 링크를 W&B에 표시합니다. 링크를 클릭하여 W&B UI에서 run을 확인합니다.
{{% /tab %}}
{{% tab "Git 저장소에서" %}}
[W&B Launch jobs 저장소의 소스 코드](https://github.com/wandb/launch-jobs)에서 동일한 hello-world job을 시작하려면 다음 코맨드를 실행합니다.

```bash
wandb launch --uri https://github.com/wandb/launch-jobs.git \\
--job-name hello-world-git --project launch-quickstart \\ 
--build-context jobs/hello_world --dockerfile Dockerfile.wandb \\ 
--entry-point "python job.py"
```
이 코맨드는 다음을 수행합니다.
1. [W&B Launch jobs 저장소](https://github.com/wandb/launch-jobs)를 임시 디렉토리에 복제합니다.
2. **hello** 프로젝트에 **hello-world-git**이라는 job을 만듭니다. 이 job은 코드를 실행하는 데 사용된 정확한 소스 코드와 설정을 추적합니다.
3. `jobs/hello_world` 디렉토리와 `Dockerfile.wandb`에서 컨테이너 이미지를 빌드합니다.
4. 컨테이너를 시작하고 `job.py` Python 스크립트를 실행합니다.

콘솔 출력은 이미지 빌드 및 실행을 보여줍니다. 컨테이너의 출력은 이전 예제와 거의 동일해야 합니다.

{{% /tab %}}
{{% tab "로컬 소스 코드에서" %}}

Git 저장소에서 버전 관리되지 않은 코드는 `--uri` 인수에 로컬 디렉토리 경로를 지정하여 시작할 수 있습니다.

빈 디렉토리를 만들고 다음 내용으로 `train.py`라는 Python 스크립트를 추가합니다.

```python
import wandb

with wandb.init() as run:
    run.log({"hello": "world"})
```

다음 내용으로 `requirements.txt` 파일을 추가합니다.

```text
wandb>=0.17.1
```

디렉토리 내에서 다음 코맨드를 실행합니다.

```bash
wandb launch --uri . --job-name hello-world-code --project launch-quickstart --entry-point "python train.py"
```

이 코맨드는 다음을 수행합니다.
1. 현재 디렉토리의 내용을 Code Artifact로 W&B에 기록합니다.
2. **launch-quickstart** 프로젝트에 **hello-world-code**라는 job을 만듭니다.
3. `train.py`와 `requirements.txt`를 기본 이미지에 복사하고 요구 사항을 `pip install`하여 컨테이너 이미지를 빌드합니다.
4. 컨테이너를 시작하고 `python train.py`를 실행합니다.
{{% /tab %}}
{{< /tabpane >}}

## 큐 만들기

Launch는 팀이 공유 컴퓨팅을 중심으로 워크플로우를 구축하는 데 도움이 되도록 설계되었습니다. 지금까지의 예제에서 `wandb launch` 코맨드는 로컬 머신에서 컨테이너를 동기적으로 실행했습니다. Launch queues 및 agents를 사용하면 공유 리소스에서 작업을 비동기적으로 실행하고 우선 순위 지정 및 하이퍼파라미터 최적화와 같은 고급 기능을 사용할 수 있습니다. 기본 큐를 만들려면 다음 단계를 따르세요.

1. [wandb.ai/launch](https://wandb.ai/launch)로 이동하여 **큐 만들기** 버튼을 클릭합니다.
2. 큐와 연결할 **Entity**를 선택합니다.
3. **큐 이름**을 입력합니다.
4. **리소스**로 **Docker**를 선택합니다.
5. 지금은 **구성**을 비워 둡니다.
6. **큐 만들기** :rocket:를 클릭합니다.

버튼을 클릭하면 브라우저가 큐 보기의 **Agents** 탭으로 리디렉션됩니다. agent가 폴링을 시작할 때까지 큐는 **활성 상태 아님** 상태로 유지됩니다.

{{< img src="/images/launch/create_docker_queue.gif" alt="" >}}

고급 큐 구성 옵션은 [고급 큐 설정 페이지]({{< relref path="./set-up-launch/setup-queue-advanced.md" lang="ko" >}})를 참조하세요.

## agent를 큐에 연결

큐에 폴링 agents가 없으면 큐 보기 화면 상단의 빨간색 배너에 **agent 추가** 버튼이 표시됩니다. 버튼을 클릭하여 agent를 실행할 코맨드를 복사하여 봅니다. 코맨드는 다음과 같습니다.

```bash
wandb launch-agent --queue <queue-name> --entity <entity-name>
```

터미널에서 코맨드를 실행하여 agent를 시작합니다. agent는 실행할 작업을 위해 지정된 큐를 폴링합니다. 수신되면 agent는 `wandb launch` 코맨드가 로컬에서 실행된 것처럼 job에 대한 컨테이너 이미지를 다운로드하거나 빌드한 다음 실행합니다.

[Launch 페이지](https://wandb.ai/launch)로 돌아가서 큐가 이제 **활성**으로 표시되는지 확인합니다.

## 큐에 job 제출

W&B 계정에서 새 **launch-quickstart** 프로젝트로 이동하여 화면 왼쪽에 있는 탐색에서 jobs 탭을 엽니다.

**Jobs** 페이지에는 이전에 실행된 runs에서 생성된 W&B Jobs 목록이 표시됩니다. launch job을 클릭하여 소스 코드, 종속성 및 job에서 생성된 runs을 확인합니다. 이 워크스루를 완료하면 목록에 세 개의 jobs이 있어야 합니다.

새 jobs 중 하나를 선택하고 다음 지침에 따라 큐에 제출합니다.

1. **Launch** 버튼을 클릭하여 job을 큐에 제출합니다. **Launch** 드로어가 나타납니다.
2. 이전에 만든 **Queue**를 선택하고 **Launch**를 클릭합니다.

이렇게 하면 job이 큐에 제출됩니다. 이 큐를 폴링하는 agent가 job을 선택하여 실행합니다. job의 진행 상황은 W&B UI에서 모니터링하거나 터미널에서 agent의 출력을 검사하여 모니터링할 수 있습니다.

`wandb launch` 코맨드는 `--queue` 인수를 지정하여 job을 큐에 직접 푸시할 수 있습니다. 예를 들어 hello-world 컨테이너 job을 큐에 제출하려면 다음 코맨드를 실행합니다.

```bash
wandb launch --docker-image wandb/job_hello_world:main --project launch-quickstart --queue <queue-name>
```

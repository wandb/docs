---
description: Easily scale and manage ML jobs using W&B Launch.
slug: /guides/launch
displayed_sidebar: default
---
import Translate, {translate} from '@docusaurus/Translate';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

# 론치

<CTAButtons colabLink="https://colab.research.google.com/drive/1wX0OSVxZJDHRsZaOaOEDx-lLUrO1hHgP"/>

W&B 론치를 사용하면 데스크탑에서 Amazon SageMaker, Kubernetes 등과 같은 컴퓨팅 자원으로 학습 [실행](../runs/intro.md)을 쉽게 확장할 수 있습니다. W&B 론치가 구성되면 몇 번의 클릭과 명령으로 학습 스크립트, 모델 평가 스위트를 빠르게 실행하고, 모델을 프로덕션 추론에 준비할 수 있습니다.

## 작동 방식

론치는 세 가지 기본 구성 요소로 구성됩니다: **론치 작업**, **큐**, 그리고 **에이전트**.

[*론치 작업*](./launch-terminology.md#launch-job)은 ML 워크플로에서 작업을 구성하고 실행하기 위한 청사진입니다. 론치 작업이 있으면, 그것을 [*론치 큐*](./launch-terminology.md#launch-queue)에 추가할 수 있습니다. 론치 큐는 선입선출(FIFO) 큐로, 특정 컴퓨팅 타깃 자원(예: Amazon SageMaker 또는 Kubernetes 클러스터)에 작업을 구성하고 제출할 수 있습니다.

작업이 큐에 추가되면 하나 이상의 [*론치 에이전트*](./launch-terminology.md#launch-agent)가 그 큐를 폴링하고 큐에 의해 타깃된 시스템에서 작업을 실행합니다.

![](/images/launch/launch_overview.png)

사용 사례에 따라, 컴퓨팅 자원 타깃을 선택한 대로 론치 큐를 구성하고 자체 인프라에 론치 에이전트를 배포해야 합니다(예: Amazon SageMaker).

[용어 및 개념](./launch-terminology.md) 페이지에서 론치 작업, 큐 작동 방식, 론치 에이전트 및 W&B 론치 작동 방식에 대한 추가 정보를 확인하세요.

## 시작 방법

사용 사례에 따라 W&B 론치를 시작하기 위해 다음 리소스를 탐색하세요:

* W&B 론치를 처음 사용하는 경우, [워크스루](./walkthrough.md) 가이드를 참고하는 것이 좋습니다.
* [W&B 론치](./setup-launch.md) 설정 방법을 알아보세요.
* [론치 작업](./create-launch-job.md)을 생성하세요.
* [Triton에 배포](https://github.com/wandb/launch-jobs/tree/main/jobs/deploy_to_nvidia_triton), [LLM 평가](https://github.com/wandb/launch-jobs/tree/main/jobs/openai_evals) 등 일반적인 작업 템플릿이 있는 W&B 론치 [공개 작업 GitHub 저장소](https://github.com/wandb/launch-jobs)를 확인하세요.
    * 이 저장소에서 생성된 론치 작업을 W&B [`wandb/jobs` 프로젝트](https://wandb.ai/wandb/jobs/jobs)에서 확인할 수 있습니다.
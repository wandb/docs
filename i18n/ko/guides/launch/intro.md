---
description: Easily scale and manage ML jobs using W&B Launch.
slug: /guides/launch
displayed_sidebar: default
---
import Translate, {translate} from '@docusaurus/Translate';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

# 시작하기

<CTAButtons colabLink="https://colab.research.google.com/drive/1wX0OSVxZJDHRsZaOaOEDx-lLUrO1hHgP"/>

Amazon SageMaker, Kubernetes 등과 같은 컴퓨팅 리소스로 데스크탑에서 트레이닝 run을 쉽게 확장할 수 있는 W&B Launch. W&B Launch가 설정되면, 몇 번의 클릭과 코맨드를 통해 트레이닝 스크립트 실행, 모델 평가, 프로덕션 추론을 위한 모델 준비 등을 빠르게 수행할 수 있습니다.

## 작동 방식

Launch는 세 가지 기본 구성 요소로 구성됩니다: **launch 작업**, **큐**, 그리고 **에이전트**.

[*launch 작업*](./launch-terminology.md#launch-job)은 ML 워크플로우에서 작업을 구성하고 실행하기 위한 청사진입니다. launch 작업이 있으면, 이를 [*launch 큐*](./launch-terminology.md#launch-queue)에 추가할 수 있습니다. launch 큐는 선입선출(FIFO) 큐로, Amazon SageMaker 또는 Kubernetes 클러스터와 같은 특정 컴퓨트 타깃 리소스에 작업을 구성하고 제출할 수 있습니다.

작업이 큐에 추가되면 하나 이상의 [*launch 에이전트*](./launch-terminology.md#launch-agent)가 해당 큐를 폴링하고 큐에 의해 타깃된 시스템에서 작업을 실행합니다.

![](/images/launch/launch_overview.png)

유스 케이스에 따라, Amazon SageMaker와 같이 선택한 [컴퓨트 리소스 타깃](./launch-terminology.md#target-resources)에 따라 launch 큐를 구성하고 자체 인프라에 launch 에이전트를 배포할 것입니다.


launch 작업, 큐 작동 방식, launch 에이전트 및 W&B Launch 작동 방식에 대한 추가 정보는 [용어 및 개념](./launch-terminology.md) 페이지를 참조하십시오.

## 시작 방법

유스 케이스에 따라 W&B Launch를 시작하기 위해 다음 리소스를 탐색하십시오:

* W&B Launch를 처음 사용하는 경우, [워크스루](./walkthrough.md) 가이드를 통해 시작하는 것이 좋습니다.
* [W&B Launch 설정](./setup-launch.md) 방법을 알아보십시오.
* [launch 작업 생성](./create-launch-job.md).
* [Triton에 배포](https://github.com/wandb/launch-jobs/tree/main/jobs/deploy_to_nvidia_triton), [LLM 평가](https://github.com/wandb/launch-jobs/tree/main/jobs/openai_evals)와 같은 일반적인 작업에 대한 템플릿이 있는 W&B Launch [공개 작업 GitHub 저장소](https://github.com/wandb/launch-jobs)를 확인하세요.
    * 이 저장소에서 생성된 launch 작업을 W&B [`wandb/jobs` 프로젝트](https://wandb.ai/wandb/jobs/jobs)에서 확인할 수 있습니다.
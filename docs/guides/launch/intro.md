---
title: Launch
description: W&B Launch를 사용하여 ML 작업을 손쉽게 확장하고 관리하세요.
slug: /guides/launch
displayed_sidebar: default
---
import Translate, {translate} from '@docusaurus/Translate';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

<CTAButtons colabLink="https://colab.research.google.com/drive/1wX0OSVxZJDHRsZaOaOEDx-lLUrO1hHgP"/>

데스크탑에서 Amazon SageMaker, Kubernetes 등의 컴퓨팅 자원으로 W&B Launch를 활용하여 손쉽게 트레이닝 [runs](../runs/intro.md)를 확장할 수 있습니다. W&B Launch가 설정되면, 몇 번의 클릭과 코맨드만으로 트레이닝 스크립트 실행, 모델 평가, 프로덕션 추론을 위한 모델 준비 등을 빠르게 수행할 수 있습니다.

## 작동 방식

Launch는 **launch jobs**, **queues**, **agents**라는 세 가지 기본 요소로 구성됩니다.

[*launch job*](./launch-terminology.md#launch-job)은 ML 워크플로우에서 작업을 구성하고 실행하기 위한 청사진입니다. launch job을 가지고 있다면, 이를 [*launch queue*](./launch-terminology.md#launch-queue)에 추가할 수 있습니다. launch queue는 FIFO(선입선출) 방식의 큐로, 특정 컴퓨팅 타겟 자원(예: Amazon SageMaker나 Kubernetes 클러스터)에 작업을 구성하고 제출할 수 있습니다.

작업이 큐에 추가되면, 하나 이상의 [*launch agents*](./launch-terminology.md#launch-agent)가 해당 큐를 폴링하여 큐가 타겟팅한 시스템에서 작업을 실행합니다.

![](/images/launch/launch_overview.png)

유스 케이스에 따라, 여러분(또는 팀의 다른 사람)은 선택한 [컴퓨팅 자원 타겟](./launch-terminology.md#target-resources)(예: Amazon SageMaker)에 맞게 launch queue를 구성하고 여러분의 인프라에 launch agent를 배포할 것입니다.

Launch jobs, 큐의 작동 방식, launch agents 및 W&B Launch의 작동 방식에 대한 추가 정보를 원하시면 [Terms and concepts](./launch-terminology.md) 페이지를 확인하세요.

## 시작 방법

유스 케이스에 따라, W&B Launch를 시작하기 위한 다음의 자료를 탐색해 보세요:

* W&B Launch를 처음 사용하시는 경우, [Walkthrough](./walkthrough.md) 가이드를 통해 진행하시길 추천드립니다.
* [W&B Launch](./setup-launch.md) 설정하는 방법을 배우세요.
* [launch job](./create-launch-job.md)을 생성하세요.
* [공개 jobs GitHub 저장소](https://github.com/wandb/launch-jobs)에서 Triton에 [배포](https://github.com/wandb/launch-jobs/tree/main/jobs/deploy_to_nvidia_triton)하기, LLM [평가](https://github.com/wandb/launch-jobs/tree/main/jobs/openai_evals)하기 등의 일반적 작업 템플릿을 확인하세요.
    * 이 저장소로부터 생성된 launch jobs는 이 공개 [`wandb/jobs` 프로젝트](https://wandb.ai/wandb/jobs/jobs) W&B 프로젝트에서 확인할 수 있습니다.
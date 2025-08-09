---
title: Azure OpenAI 파인튜닝
description: W&B로 Azure OpenAI 모델을 파인튜닝하는 방법
menu:
  default:
    identifier: ko-guides-integrations-azure-openai-fine-tuning
    parent: integrations
weight: 20
---

## 소개
Microsoft Azure에서 W&B를 활용해 GPT-3.5 또는 GPT-4 모델을 파인튜닝하면, 메트릭을 자동으로 캡처하고 W&B의 실험 추적 및 평가 툴을 통해 모델 성능을 체계적으로 평가하고 향상시킬 수 있습니다.

{{< img src="/images/integrations/aoai_ft_plot.png" alt="Azure OpenAI 파인튜닝 메트릭" >}}

## 사전 준비 사항
- [공식 Azure 문서](https://wandb.me/aoai-wb-int)를 참고하여 Azure OpenAI 서비스를 설정하세요.
- API 키가 포함된 W&B 계정을 설정하세요.

## 워크플로우 개요

### 1. 파인튜닝 설정
- Azure OpenAI 요구사항에 맞게 트레이닝 데이터를 준비하세요.
- Azure OpenAI에서 파인튜닝 작업을 설정하세요.
- W&B는 파인튜닝 프로세스를 자동으로 추적하며, 메트릭과 하이퍼파라미터를 로그로 남깁니다.

### 2. Experiment 추적
파인튜닝 도중, W&B는 다음을 캡처합니다:
- 트레이닝 및 검증 메트릭
- 모델 하이퍼파라미터
- 리소스 사용량
- 트레이닝 Artifacts

### 3. 모델 평가
파인튜닝이 끝나면 [W&B Weave](https://weave-docs.wandb.ai)를 활용해 다음을 진행할 수 있습니다:
- 모델 출력값을 참조 데이터셋과 비교 평가
- 여러 파인튜닝 run의 성능 비교
- 특정 테스트 케이스에서의 모델 행동 분석
- 데이터 기반의 모델 선택 결정

## 실제 예시
* [의료 노트 생성 데모](https://wandb.me/aoai-ft-colab)를 참고하면, 이 인테그레이션을 통해 다음이 어떻게 가능한지 볼 수 있습니다:
  - 파인튜닝 Experiment의 체계적인 추적
  - 도메인 특화 메트릭을 활용한 모델 평가
* [노트북에서 파인튜닝하는 인터랙티브 데모](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/azure/azure_gpt_medical_notes.ipynb)도 살펴보세요.

## 추가 자료
- [Azure OpenAI W&B Integration Guide](https://wandb.me/aoai-wb-int)
- [Azure OpenAI 파인튜닝 공식 문서](https://learn.microsoft.com/azure/ai-services/openai/how-to/fine-tuning?tabs=turbo%2Cpython&pivots=programming-language-python)
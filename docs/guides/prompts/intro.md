---
description: Tools for the development of LLM-powered applications
slug: /guides/prompts
displayed_sidebar: default
---

# LLM을 위한 프롬프트

W&B 프롬프트는 LLM 기반 애플리케이션 개발을 위해 구축된 LLMOps 도구 모음입니다. W&B 프롬프트를 사용하여 LLM의 실행 흐름을 시각화하고 검사하고, LLM의 입력 및 출력을 분석하고, 중간 결과를 확인하며 프롬프트와 LLM 체인 구성을 안전하게 저장하고 관리하세요.

## 사용 사례

W&B 프롬프트는 LLM 기반 앱을 구축하고 평가하기 위한 해결책입니다. 소프트웨어 개발자, 프롬프트 엔지니어, 기계학습 실무자, 데이터 과학자 및 LLM을 사용하는 다른 이해관계자들은 LLM 체인과 프롬프트를 더 세밀하게 탐색하고 디버깅하기 위해 첨단 도구가 필요합니다.

- LLM 애플리케이션의 입력 및 출력 추적
- 대화형 추적을 사용하여 LLM 체인 및 프롬프트 디버깅
- LLM 체인 및 프롬프트의 성능 평가

## 제품

### 추적

W&B의 LLM 도구는 *추적*이라고 합니다. **추적**을 사용하면 LLM 체인의 입력 및 출력, 실행 흐름, 모델 아키텍처 및 중간 결과를 추적하고 시각화할 수 있습니다.

LLM 체이닝, 플러그인 또는 파이프라이닝 사용 사례에 추적을 사용하세요. 자체 LLM 체이닝 구현을 사용하거나 LangChain과 같은 LLM 라이브러리에서 제공하는 W&B 통합을 사용할 수 있습니다.

추적에는 세 가지 주요 구성 요소가 있습니다:

- [추적 테이블](https://docs.wandb.ai/guides/prompts#trace-table): 체인의 입력 및 출력 개요.
- [추적 타임라인](https://docs.wandb.ai/guides/prompts#trace-timeline): 체인의 실행 흐름을 표시하며 구성 요소 유형에 따라 색상으로 구분됩니다.
- [모델 아키텍처](https://docs.wandb.ai/guides/prompts#model-architecture): 체인의 구조와 체인의 각 구성 요소를 초기화하는 데 사용된 파라미터에 대한 세부 정보를 볼 수 있습니다.

**추적 테이블**

추적 테이블은 체인의 입력 및 출력에 대한 개요를 제공합니다. 추적 테이블은 또한 체인에서 추적 이벤트의 구성, 체인이 성공적으로 실행되었는지 여부 및 체인을 실행할 때 반환된 오류 메시지에 대한 정보도 제공합니다.

![추적 테이블의 스크린샷.](/images/prompts/trace_table.png)

테이블의 왼쪽에 있는 행 번호를 클릭하면 해당 체인 인스턴스의 추적 타임라인을 볼 수 있습니다.

**추적 타임라인**

추적 타임라인 뷰는 구성 요소 유형에 따라 색상으로 구분된 체인의 실행 흐름을 표시합니다. 추적 이벤트를 선택하여 해당 추적의 입력, 출력 및 메타데이터를 표시합니다.

![추적 타임라인의 스크린샷.](/images/prompts/trace_timeline.png)

오류를 발생시킨 추적 이벤트는 빨간색으로 테두리가 그려집니다. 빨간색으로 표시된 추적 이벤트를 클릭하면 반환된 오류 메시지를 볼 수 있습니다.

![추적 타임라인 오류의 스크린샷.](/images/prompts/trace_timeline_error.png)

**모델 아키텍처**

모델 아키텍처 뷰는 체인의 구조와 체인의 각 구성 요소를 초기화하는 데 사용된 파라미터에 대한 세부 정보를 제공합니다. 추적 이벤트를 클릭하여 해당 이벤트에 대한 자세한 내용을 알아보세요.

**평가**

애플리케이션이 개선되고 있는지 평가할 방법이 필요합니다. 이를 위해 일반적인 관행은 변경이 있을 때마다 동일한 데이터세트에 대해 테스트하는 것입니다. W&B를 사용하여 LLM 애플리케이션을 평가하는 방법을 배우려면 이 튜토리얼을 참조하세요.
[튜토리얼: LLM 애플리케이션 성능 평가](https://github.com/wandb/examples/blob/master/colabs/prompts/prompts_evaluation.ipynb)

## 통합

Weights and Biases는 다음에 대한 경량 통합도 제공합니다:

- [LangChain](https://docs.wandb.ai/guides/integrations/langchain)
- [OpenAI API](https://docs.wandb.ai/guides/integrations/openai-api)
- [OpenAI GPT-3.5 파인 튜닝](https://docs.wandb.ai/guides/integrations/openai)
- [Hugging Face Transformers](https://docs.wandb.ai/guides/integrations/huggingface)

## 시작하기

추적을 사용하여 사용자 정의 LLM 파이프라인을 로깅하는 방법을 안내하는 프롬프트 [퀵스타트](https://docs.wandb.ai/guides/prompts/quickstart) 가이드를 확인하는 것이 좋습니다. 가이드의 [colab](http://wandb.me/prompts-quickstart) 버전도 사용할 수 있습니다.

## 다음 단계

- [추적](https://colab.research.google.com/github/wandb/weave/blob/master/examples/prompts/trace_debugging/trace_quickstart_langchain.ipynb)에 대한 더 자세한 문서나 [OpenAI](https://docs.wandb.ai/guides/prompts/openai) 통합을 확인하세요.
- LLMOps에 프롬프트를 사용하는 방법에 대한 더 자세한 설명을 제공하는 [데모 colabs](https://github.com/wandb/examples/tree/master/colabs/prompts) 중 하나를 시도해 보세요.
- 다음 튜토리얼을 참조하여 LLM 애플리케이션 성능을 추적하는 기존 W&B 기능인 Tables 및 Runs를 사용할 수 있습니다:
[튜토리얼: LLM 애플리케이션 성능 평가](https://github.com/wandb/examples/blob/master/colabs/prompts/prompts_evaluation.ipynb)
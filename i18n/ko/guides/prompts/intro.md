---
description: Tools for the development of LLM-powered applications
slug: /guides/prompts
displayed_sidebar: default
---

# LLM을 위한 프롬프트

W&B 프롬프트는 LLM 기반 애플리케이션 개발을 위해 구축된 LLMOps 툴 모음입니다. W&B 프롬프트를 사용하여 LLM의 실행 흐름을 시각화하고 검사하고, LLM의 입력과 출력을 분석하며, 중간 결과를 확인하고 프롬프트 및 LLM 체인 설정을 안전하게 저장하고 관리하세요.

## 유스 케이스

W&B 프롬프트는 LLM 기반 앱을 구축하고 평가하기 위한 해결책입니다. 소프트웨어 개발자, 프롬프트 엔지니어, ML 실무자, 데이터 과학자 및 LLM을 사용하는 기타 이해 관계자들은 LLM 체인과 프롬프트를 더 세밀하게 탐색하고 디버깅하기 위해 최첨단의 도구가 필요합니다.

- LLM 애플리케이션의 입력 및 출력 추적
- 인터랙티브 추적을 사용하여 LLM 체인과 프롬프트 디버깅
- LLM 체인 및 프롬프트의 성능 평가

## 제품

### 추적(Traces)

W&B의 LLM 툴은 *추적(Traces)*이라고 불립니다. **추적**은 LLM 체인의 입력과 출력, 실행 흐름, 모델 아키텍처 및 모든 중간 결과를 추적하고 시각화할 수 있게 해줍니다.

LLM 체이닝, 플러그인 또는 파이프라이닝 유스 케이스에 대해 추적을 사용하세요. 자체 LLM 체이닝 구현을 사용하거나 LangChain과 같은 LLM 라이브러리에서 제공하는 W&B 인테그레이션을 사용할 수 있습니다.

추적은 세 가지 주요 구성 요소로 구성됩니다:

- [추적 테이블](https://docs.wandb.ai/guides/prompts#trace-table): 체인의 입력과 출력에 대한 개요.
- [추적 타임라인](https://docs.wandb.ai/guides/prompts#trace-timeline): 체인의 실행 흐름을 표시하며 구성 요소 유형에 따라 색상으로 코드화됩니다.
- [모델 아키텍처](https://docs.wandb.ai/guides/prompts#model-architecture): 체인의 구조와 체인의 각 구성 요소를 초기화하는 데 사용된 파라미터에 대한 세부 정보를 봅니다.

**추적 테이블**

추적 테이블은 체인의 입력과 출력에 대한 개요를 제공합니다. 추적 테이블은 또한 체인에서 추적 이벤트의 구성, 체인이 성공적으로 실행되었는지 여부 및 체인을 실행할 때 반환된 오류 메시지에 대한 정보도 제공합니다.

![추적 테이블의 스크린샷입니다.](/images/prompts/trace_table.png)

테이블 왼쪽에 있는 행 번호를 클릭하여 해당 체인 인스턴스의 추적 타임라인을 확인하세요.

**추적 타임라인**

추적 타임라인 뷰는 체인의 실행 흐름을 표시하며 구성 요소 유형에 따라 색상으로 코드화됩니다. 추적 이벤트를 선택하면 해당 추적의 입력, 출력 및 메타데이터가 표시됩니다.

![추적 타임라인의 스크린샷입니다.](/images/prompts/trace_timeline.png)

오류를 발생시키는 추적 이벤트는 빨간색으로 표시됩니다. 빨간색으로 표시된 추적 이벤트를 클릭하면 반환된 오류 메시지를 확인할 수 있습니다.

![추적 타임라인 오류의 스크린샷입니다.](/images/prompts/trace_timeline_error.png)

**모델 아키텍처**

모델 아키텍처 뷰는 체인의 구조와 체인의 각 구성 요소를 초기화하는 데 사용된 파라미터에 대한 세부 정보를 제공합니다. 추적 이벤트를 클릭하여 해당 이벤트에 대한 자세한 정보를 알아보세요.

**평가** 

애플리케이션이 개선되고 있는지 평가하기 위해서는 방법이 필요합니다. 이를 위한 일반적인 관행은 변경이 있을 때 동일한 데이터셋에 대해 테스트하는 것입니다. W&B를 사용하여 LLM 애플리케이션을 평가하는 방법을 알아보려면 이 튜토리얼을 참조하세요.
[튜토리얼: LLM 애플리케이션 성능 평가](https://github.com/wandb/examples/blob/master/colabs/prompts/prompts_evaluation.ipynb)

## 인테그레이션

Weights and Biases는 다음에 대한 가벼운 인테그레이션도 제공합니다:

- [LangChain](https://docs.wandb.ai/guides/integrations/langchain)
- [OpenAI API](https://docs.wandb.ai/guides/integrations/openai-api)
- [OpenAI GPT-3.5 파인튜닝](https://docs.wandb.ai/guides/integrations/openai)
- [Hugging Face 트랜스포머](https://docs.wandb.ai/guides/integrations/huggingface)

## 시작하기

Prompts [퀵스타트](https://docs.wandb.ai/guides/prompts/quickstart) 가이드를 참조하는 것이 좋습니다. 이 가이드는 Trace로 사용자 정의 LLM 파이프라인을 로깅하는 방법을 안내합니다. 가이드의 [colab](http://wandb.me/prompts-quickstart) 버전도 이용 가능합니다.

## 다음 단계

- [Trace](https://colab.research.google.com/github/wandb/weave/blob/master/examples/prompts/trace_debugging/trace_quickstart_langchain.ipynb)에 대한 더 자세한 문서나 [OpenAI](https://docs.wandb.ai/guides/prompts/openai) 인테그레이션을 확인하세요.
- LLMOps에 대한 Prompts 사용 방법에 대한 더 자세한 설명을 제공하는 [데모 colabs](https://github.com/wandb/examples/tree/master/colabs/prompts) 중 하나를 시도해 보세요.
- 기존 W&B 기능인 Tables와 Runs를 사용하여 LLM 애플리케이션 성능을 추적할 수 있습니다. 자세한 내용을 알아보려면 이 튜토리얼을 참조하세요:
[튜토리얼: LLM 애플리케이션 성능 평가](https://github.com/wandb/examples/blob/master/colabs/prompts/prompts_evaluation.ipynb)
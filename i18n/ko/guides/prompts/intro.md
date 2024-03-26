---
description: Tools for the development of LLM-powered applications
slug: /guides/prompts
displayed_sidebar: default
---

# LLM을 위한 Prompts

W&B Prompts는 LLM 기반 애플리케이션 개발을 위해 구축된 LLMOps 툴의 모음입니다. W&B Prompts를 사용하여 LLM의 실행 흐름을 시각화 및 검사하고, LLM의 입력과 출력을 분석하며, 중간 결과를 확인하고 프롬프트 및 LLM 체인 구성을 안전하게 저장하고 관리할 수 있습니다.

## 유스 케이스

W&B Prompts는 LLM 기반 앱을 구축하고 평가하기 위한 솔루션입니다. 소프트웨어 개발자, 프롬프트 엔지니어, ML 실무자, 데이터사이언티스트 및 LLM을 사용하는 기타 이해 관계자들은 LLM 체인과 프롬프트를 더 세밀하게 탐색하고 디버깅할 수 있는 최첨단의 도구가 필요합니다.

- LLM 애플리케이션의 입력 및 출력 추적
- 인터랙티브 추적을 통한 LLM 체인과 프롬프트 디버깅
- LLM 체인 및 프롬프트의 성능 평가

## 제품

### Traces

W&B의 LLM 도구는 *Traces*라고 합니다. **Traces**는 LLM 체인의 입력과 출력, 실행 흐름, 모델 아키텍처 및 모든 중간 결과를 추적하고 시각화할 수 있게 해줍니다.

LLM 체이닝, 플러그인 또는 파이프라이닝 유스 케이스에 Traces을 사용하세요. 자체 LLM 체이닝 구현을 사용하거나 LangChain과 같은 LLM 라이브러리에서 제공하는 W&B 인테그레이션을 사용할 수 있습니다.

Traces은 세 가지 주요 구성 요소로 이루어져 있습니다:

- [Trace Table](https://docs.wandb.ai/guides/prompts#trace-table): 체인의 입력과 출력 개요를 볼 수 있습니다. 
- [Trace Timeline](https://docs.wandb.ai/guides/prompts#trace-timeline): 체인의 실행 흐름을 표시하며 구성 요소 유형에 따라 다른 색상으로 구분합니다.
- [Model Architecture](https://docs.wandb.ai/guides/prompts#model-architecture): 체인의 구조와 체인의 각 구성 요소를 초기화하는 데 사용된 파라미터에 대한 세부 정보를 확인할 수 있습니다.

**Trace Table**

Trace Table은 체인의 입력과 출력에 대한 개요를 제공합니다. 또한 Trace Table은 체인에서 추적 이벤트의 구성, 체인의 성공적으로 실행 여부 및 체인 실행 시 반환된 오류 메시지에 대한 정보도 제공합니다.

![Trace Table의 스크린샷입니다.](/images/prompts/trace_table.png)

Table 왼쪽에 있는 행 번호를 클릭하여 해당 체인 인스턴스의 Trace Timeline을 확인하세요.

**Trace Timeline**

Trace Timeline 뷰는 체인의 실행 흐름을 표시하며 구성 요소 유형에 따라 다른 색상으로 구분합니다. 추적 이벤트를 선택하면 해당 추적의 입력, 출력 및 메타데이터가 표시됩니다.

![Trace Timeline의 스크린샷입니다.](/images/prompts/trace_timeline.png)

오류를 발생시키는 추적 이벤트는 빨간색으로 표시됩니다. 빨간색으로 표시된 추적 이벤트를 클릭하면 반환된 오류 메시지를 확인할 수 있습니다.

![Trace Timeline 오류의 스크린샷입니다.](/images/prompts/trace_timeline_error.png)

**Model Architecture**

Model Architecture 뷰는 체인의 구조와 체인의 각 구성 요소를 초기화하는 데 사용된 파라미터에 대한 세부 정보를 제공합니다. 추적 이벤트를 클릭하면 해당 이벤트에 대한 자세한 확인할 수 있습니다.

**평가** 

애플리케이션을 반복하려면 애플리케이션이 개선되고 있는지 평가하는 방법이 필요합니다. 이를 위해서는 변경 사항이 있을 때 동일한 데이터 세트와 비교하여 테스트하는 것이 일반적인 방법입니다. W&B를 사용하여 LLM 애플리케이션을 평가하는 방법을 알아보려면 이 튜토리얼을 참조하세요.
[튜토리얼: LLM 애플리케이션 성능 평가](https://github.com/wandb/examples/blob/master/colabs/prompts/prompts_evaluation.ipynb)

## 인테그레이션

Weights and Biases는 다음과 같은 가벼운 인테그레이션도 제공합니다:

- [LangChain](https://docs.wandb.ai/guides/integrations/langchain)
- [OpenAI API](https://docs.wandb.ai/guides/integrations/openai-api)
- [OpenAI GPT-3.5 파인튜닝](https://docs.wandb.ai/guides/integrations/openai)
- [Hugging Face 트랜스포머](https://docs.wandb.ai/guides/integrations/huggingface)

## 시작하기

Prompts [퀵스타트](https://docs.wandb.ai/guides/prompts/quickstart) 가이드에서는 Trace를 사용하여 커스텀 LLM 파이프라인을 로깅하는 방법을 안내해 드립니다. 가이드의 [colab](http://wandb.me/prompts-quickstart) 버전도 제공하여 드리고 있습니다.

## 다음 단계

- [Trace](https://colab.research.google.com/github/wandb/weave/blob/master/examples/prompts/trace_debugging/trace_quickstart_langchain.ipynb)에 대한 더 자세한 문서나 [OpenAI](https://docs.wandb.ai/guides/prompts/openai) Integration을 읽어보세요.
- LLMOps용 Prompts 사용 방법에 대해 더 자세히 설명하는 [데모 colabs](https://github.com/wandb/examples/tree/master/colabs/prompts) 중 하나를 시도해 보세요.
- Tables와 Runs같은 기존 W&B 기능을 사용하여 LLM 애플리케이션 성능을 추적할 수 있습니다. 자세한 내용은 이 튜토리얼을 참조하세요:
[튜토리얼: LLM 애플리케이션 성능 평가](https://github.com/wandb/examples/blob/master/colabs/prompts/prompts_evaluation.ipynb)
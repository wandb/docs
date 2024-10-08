---
title: Prompts for LLMs
description: LLM 기반 애플리케이션 개발을 위한 툴
slug: /guides/integrations/prompts
displayed_sidebar: default
---
import { WEAVE_DOCS_URL } from '@site/src/util/links';

<a href={WEAVE_DOCS_URL} target="_blank">
    <img className="no-zoom" src="/images/weave/weave_banner.png" alt="Building LLM apps? Try Weave" style={{display: "block", marginBottom: "15px"}} />
</a>

## 프롬프트

W&B 프롬프트는 LLM 기반 애플리케이션 개발을 위한 LLMOps 툴 모음입니다. W&B 프롬프트를 사용하여 LLM의 실행 흐름을 시각화하고 검사하며, LLM의 입력 및 출력을 분석하고, 중간 결과를 확인하며, 프롬프트 및 LLM 체인 설정을 안전하게 저장하고 관리할 수 있습니다.

## 유스 케이스

W&B 프롬프트는 LLM 기반 앱을 구축하고 평가하는 솔루션입니다. 소프트웨어 개발자, 프롬프트 엔지니어, ML 실무자, 데이터 과학자 등 LLM과 함께 작업하는 이해관계자들은 LLM 체인과 프롬프트를 보다 세세하게 탐색하고 디버그할 수 있는 최첨단의 툴이 필요합니다.

- LLM 애플리케이션의 입력과 출력을 추적하십시오.
- 상호작용 가능한 추적을 사용하여 LLM 체인과 프롬프트를 디버그하십시오.
- LLM 체인과 프롬프트의 성능을 평가하십시오.

## 제품들

### Traces

W&B의 LLM 툴은 *Traces*라고 불립니다. **Traces**는 LLM 체인의 입력과 출력, 실행 흐름, 모델 아키텍처, 그리고 중간 결과를 추적하고 시각화할 수 있게 합니다.

Traces는 LLM 체인 연결, 플러그인 또는 파이프라이닝 유스 케이스에 사용됩니다. 자체 LLM 체인 연결 구현을 사용할 수 있으며, LangChain과 같은 LLM 라이브러리에서 제공하는 W&B 인테그레이션을 사용할 수도 있습니다.

Traces는 세 가지 주요 구성 요소로 구성됩니다:

- [Trace table](#trace-table): 체인의 입력과 출력에 대한 개요입니다.
- [Trace timeline](#trace-timeline): 체인의 실행 흐름을 표시하며, 구성 요소 유형에 따라 색상으로 구분됩니다.
- [Model architecture](#model-architecture): 체인의 구조 및 각 구성 요소를 초기화하는 데 사용된 파라미터에 대한 세부 정보를 볼 수 있습니다.

#### Trace Table

Trace Table은 체인의 입력과 출력에 대한 개요를 제공합니다. Trace Table은 또한 체인 내에서 trace 이벤트의 구성, 체인이 성공적으로 실행되었는지 여부, 체인 실행 시 반환된 오류 메시지를 제공합니다.

![Screenshot of a trace table.](/images/prompts/trace_table.png)

Table의 왼쪽 행 번호를 클릭하여 해당 체인 인스턴스의 Trace Timeline을 볼 수 있습니다.

#### Trace Timeline

Trace Timeline 뷰는 체인의 실행 흐름을 표시하며, 구성 요소 유형에 따라 색상으로 구분됩니다. Trace 이벤트를 선택하여 해당 trace의 입력, 출력 및 메타데이터를 표시합니다.

![Screenshot of a Trace Timeline.](/images/prompts/trace_timeline.png)

오류를 발생시키는 trace 이벤트는 빨간색으로 표시됩니다. 빨간색으로 표시된 trace 이벤트를 클릭하여 반환된 오류 메시지를 확인하십시오.

![Screenshot of a Trace Timeline error.](/images/prompts/trace_timeline_error.png)

#### Model Architecture

Model Architecture 뷰는 체인의 구조 및 각 구성 요소를 초기화하는 데 사용된 파라미터에 대한 세부 정보를 제공합니다. Trace 이벤트를 클릭하여 해당 이벤트에 대한 더 많은 세부 정보를 확인하십시오.

## 평가

애플리케이션을 반복해 나가려면 개선되고 있는지 평가하는 방법이 필요합니다. 이를 위해 일반적인 방법은 변경 사항이 있을 때 동일한 데이터셋에 대해 테스트하는 것입니다. W&B를 사용하여 LLM 애플리케이션을 평가하는 방법을 배우려면 이 튜토리얼을 참조하십시오.
[Tutorial: Evaluate LLM application performance](https://github.com/wandb/examples/blob/master/colabs/prompts/prompts_evaluation.ipynb)

## 인테그레이션

Weights and Biases는 다음을 위한 경량 인테그레이션도 제공합니다:

- [LangChain](/guides/integrations/langchain)
- [OpenAI API](/guides/integrations/openai-api)
- [OpenAI GPT-3.5 Fine-Tuning](/guides/integrations/openai)
- [Hugging Face Transformers](/guides/integrations/huggingface)

## 시작하기

Prompts [퀵스타트](./quickstart.md) 가이드를 통해 커스텀 LLM 파이프라인을 Trace와 함께 로그로 기록하는 방법을 설명합니다. 가이드의 [colab](http://wandb.me/prompts-quickstart) 버전도 제공됩니다.

## 다음 단계

- [Trace](https://colab.research.google.com/github/wandb/weave/blob/master/examples/prompts/trace_debugging/trace_quickstart_langchain.ipynb) 및 [OpenAI](/guides/integrations/prompts/openai/) 인테그레이션에 대한 더 자세한 문서를 확인하십시오.
- LLMOps에 Prompts를 사용하는 방법에 대한 보다 자세한 설명이 포함된 우리의 [데모 colabs](https://github.com/wandb/examples/tree/master/colabs/prompts)를 시도해보십시오.
- LLM 애플리케이션 성능을 추적하는데 Tables와 Runs 같은 기존의 W&B 기능을 사용할 수 있습니다. 이 튜토리얼을 참조하여 더 알아보십시오:
[Tutorial: Evaluate LLM application performance](https://github.com/wandb/examples/blob/master/colabs/prompts/prompts_evaluation.ipynb)
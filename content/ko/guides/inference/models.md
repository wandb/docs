---
title: 사용 가능한 모델
description: W&B Inference에서 제공하는 foundation 모델을 찾아보세요
menu:
  default:
    identifier: ko-guides-inference-models
weight: 10
---

W&B Inference는 여러 오픈 소스 기반 모델에 엑세스를 제공합니다. 각 모델은 서로 다른 강점과 유스 케이스를 가지고 있습니다.

## 모델 비교

| Model | API 사용 시 Model ID | 유형 | 컨텍스트 윈도우 | 파라미터 | 설명 |
|-------|--------------------------|------|----------------|------------|-------------|
| OpenAI GPT OSS 120B	| `openai/gpt-oss-120b` | 텍스트 | 131,000 | 5.1B-117B (사용중-전체) | 고차원 추론, 에이전트, 범용 유스 케이스를 위해 설계된 효율적인 Mixture-of-Experts 모델입니다. |
| OpenAI GPT OSS 20B | `openai/gpt-oss-20b` | 텍스트 | 131,000 | 3.6B-20B (사용중-전체) | OpenAI의 Harmony 응답 형식으로 학습된, 추론 능력을 갖춘 저지연 Mixture-of-Experts 모델입니다. |
| Qwen3 235B A22B Thinking-2507 | `Qwen/Qwen3-235B-A22B-Thinking-2507` | 텍스트 | 262K | 22B-235B (사용중-전체) | 구조화된 추론, 수학, 장문 생성에 최적화된 고성능 Mixture-of-Experts 모델입니다. |
| Qwen3 235B A22B-2507 | `Qwen/Qwen3-235B-A22B-Instruct-2507` | 텍스트 | 262K | 22B-235B (사용중-전체) | 논리적 추론에 최적화된 효율적인 다국어 Mixture-of-Experts, instruction-tuned 모델입니다. |
| Qwen3 Coder 480B A35B | `Qwen/Qwen3-Coder-480B-A35B-Instruct` | 텍스트 | 262K | 35B-480B (사용중-전체) | 함수 호출, 툴 활용, 장문 추론과 같은 코딩 작업에 최적화된 Mixture-of-Experts 모델입니다. |
| MoonshotAI Kimi K2 | `moonshotai/Kimi-K2-Instruct` | 텍스트 | 128K | 32B-1T (사용중-전체) | 복잡한 툴 활용, 추론, 코드 합성에 최적화된 Mixture-of-Experts 모델입니다. |
| DeepSeek R1-0528 | `deepseek-ai/DeepSeek-R1-0528` | 텍스트 | 161K | 37B-680B (사용중-전체) | 복잡한 코딩, 수학, 구조화된 문서 분석 등 정밀한 추론 작업에 최적화되었습니다. |
| DeepSeek V3-0324 | `deepseek-ai/DeepSeek-V3-0324` | 텍스트 | 161K | 37B-680B (사용중-전체) | 고난도 언어 처리 및 종합 문서 분석에 특화된 강력한 Mixture-of-Experts 모델입니다. |
| Meta Llama 3.1 8B | `meta-llama/Llama-3.1-8B-Instruct` | 텍스트 | 128K | 8B (전체) | 빠르고 다국어 챗봇 상호작용에 최적화된 효율적인 대화형 모델입니다. |
| Meta Llama 3.3 70B | `meta-llama/Llama-3.3-70B-Instruct` | 텍스트 | 128K | 70B (전체) | 대화, 상세한 지시 수행, 코딩 작업에서 뛰어난 다국어 모델입니다. |
| Meta Llama 4 Scout | `meta-llama/Llama-4-Scout-17B-16E-Instruct` | 텍스트, 비전 | 64K | 17B-109B (사용중-전체) | 텍스트와 이미지 이해가 결합된 멀티모달 모델로, 시각 작업과 융합 분석에 이상적입니다. |
| Microsoft Phi 4 Mini 3.8B | `microsoft/Phi-4-mini-instruct` | 텍스트 | 128K | 3.8B (사용중-전체) | 자원이 제한된 환경에서 신속한 응답에 적합한 작고 효율적인 모델입니다. |

## Model ID 사용법

API를 사용할 때, 위 표의 Model ID를 지정해서 사용할 수 있습니다. 예시:

```python
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[...]
)
```

## 다음 단계

- 각 모델의 [사용량 제한 및 가격]({{< relref path="usage-limits" lang="ko" >}})을 확인하세요
- 모델 활용법은 [API reference]({{< relref path="api-reference" lang="ko" >}})를 참고하세요
- [W&B Playground]({{< relref path="ui-guide" lang="ko" >}})에서 모델을 직접 체험해보세요
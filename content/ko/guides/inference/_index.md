---
title: W&B Inference
description: W&B Weave 와 OpenAI 호환 API 를 통해 오픈소스 foundation models 에 엑세스하세요
menu:
  default:
    identifier: ko-guides-inference-_index
weight: 8
---

W&B Inference를 사용하면 W&B Weave와 OpenAI 호환 API를 통해 최신 오픈소스 파운데이션 모델에 엑세스할 수 있습니다. 다음과 같은 일을 할 수 있습니다:

- 별도의 호스팅 제공업체 가입이나 모델을 직접 호스팅할 필요 없이 AI 애플리케이션과 에이전트를 구축할 수 있습니다.
- W&B Weave Playground에서 [지원되는 모델]({{< relref path="models" lang="ko" >}})을 바로 테스트해 볼 수 있습니다.

Weave를 통해 W&B Inference 기반 애플리케이션을 추적, 평가, 모니터링 및 개선할 수 있습니다.

## 퀵스타트

아래는 Python을 사용한 간단한 예시입니다:

```python
import openai

client = openai.OpenAI(
    # 커스텀 base URL은 W&B Inference를 가리킵니다.
    base_url='https://api.inference.wandb.ai/v1',
    
    # https://wandb.ai/authorize에서 API 키를 받으세요.
    api_key="<your-api-key>"
)

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."}
    ],
)

print(response.choices[0].message.content)
```

## 다음 단계

1. [사용 가능한 모델]({{< relref path="models" lang="ko" >}}) 및 [사용 정보와 제한]({{< relref path="usage-limits" lang="ko" >}})을 확인하세요.
2. [사전 준비 사항]({{< relref path="prerequisites" lang="ko" >}})을 참고해 계정을 세팅하세요.
3. [API]({{< relref path="api-reference" lang="ko" >}}) 또는 [UI]({{< relref path="ui-guide" lang="ko" >}})를 통해 서비스를 이용해보세요.
4. [사용 예시]({{< relref path="examples" lang="ko" >}})를 시도해보세요.

## 사용 안내

{{< alert title="중요" color="warning" >}}
W&B Inference 크레딧은 Free, Pro, Academic 요금제에서 한시적으로 제공됩니다. Enterprise 계정에서는 제공 여부가 다를 수 있습니다. 크레딧이 소진되면:

- **Free 사용자**는 계속 사용하려면 유료 요금제로 업그레이드해야 합니다.  
  👉 [Pro 또는 Enterprise로 업그레이드](https://wandb.ai/subscriptions)
- **Pro 사용자**는 무료 크레딧을 초과한 이용분에 대해 월 $6,000 한도 내에서 과금됩니다. 자세한 사항은 [계정 등급 및 기본 사용 한도]({{< relref path="usage-limits#account-tiers-and-default-usage-caps" lang="ko" >}})를 참고하세요.
- **Enterprise 사용량**은 연간 $700,000 한도 내에서 제한됩니다. 과금 및 한도 증설은 담당 영업대표와 상의해야 합니다. 자세한 사항은 [계정 등급 및 기본 사용 한도]({{< relref path="usage-limits#account-tiers-and-default-usage-caps" lang="ko" >}})를 참고하세요.

자세한 정보는 [요금제 페이지](https://wandb.ai/site/pricing/) 또는 [모델별 비용](https://wandb.ai/site/pricing/inference)을 참고하세요.
{{< /alert >}}
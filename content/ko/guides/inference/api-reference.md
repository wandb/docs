---
title: API 레퍼런스
description: W&B Inference 서비스의 전체 API 레퍼런스
linkTitle: API Reference
menu:
  default:
    identifier: ko-guides-inference-api-reference
weight: 40
---

W&B Inference API를 사용해 파운데이션 모델에 프로그래밍 방식으로 엑세스하는 방법을 알아보세요.

{{< alert title="Tip" >}}
API 오류가 발생하나요? 해결 방법은 [W&B Inference 지원 문서](/support/inference/)를 참고하세요.
{{< /alert >}}

## 엔드포인트

Inference 서비스 엑세스 주소:

```plaintext
https://api.inference.wandb.ai/v1
```

{{< alert title="Important" >}}
이 엔드포인트를 이용하려면 다음이 필요합니다:
- Inference 크레딧이 있는 W&B 계정
- 유효한 W&B API 키
- 하나의 W&B Entity(팀)와 프로젝트

코드 예제에서는 `<your-team>/<your-project>`와 같이 나타납니다.
{{< /alert >}}

## 지원 메소드

Inference API는 다음 메소드를 지원합니다:

### Chat completions

`/chat/completions` 엔드포인트를 사용해 챗 컴플리션(chat completion)을 생성할 수 있습니다. 이 엔드포인트는 OpenAI 형식을 따라 메시지를 보내고 응답을 받습니다.

챗 컴플리션을 생성하려면 다음 정보를 입력하세요:
- Inference 서비스 기본 URL: `https://api.inference.wandb.ai/v1`
- 본인의 W&B API 키: `<your-api-key>`
- 본인의 W&B Entity와 프로젝트: `<your-team>/<your-project>`
- [사용 가능한 모델]({{< relref path="models" lang="ko" >}}) 중 하나의 모델 ID

{{< tabpane text=true >}}
{{% tab header="Bash" value="bash" %}}

```bash
curl https://api.inference.wandb.ai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-api-key>" \
  -H "OpenAI-Project: <your-team>/<your-project>" \
  -d '{
    "model": "<model-id>",
    "messages": [
      { "role": "system", "content": "You are a helpful assistant." },
      { "role": "user", "content": "Tell me a joke." }
    ]
  }'
```

{{% /tab %}}
{{% tab header="Python" value="python" %}}

```python
import openai

client = openai.OpenAI(
    # 커스텀 base URL을 W&B Inference로 지정
    base_url='https://api.inference.wandb.ai/v1',

    # https://wandb.ai/authorize 에서 API 키를 발급받으세요
    # 보안을 위해 환경변수 OPENAI_API_KEY로 지정하는 것도 권장합니다
    api_key="<your-api-key>"
)

# <model-id>를 사용 가능한 모델 ID로 바꿔주세요
response = client.chat.completions.create(
    model="<model-id>",
    messages=[
        {"role": "system", "content": "<your-system-prompt>"},
        {"role": "user", "content": "<your-prompt>"}
    ],
)

print(response.choices[0].message.content)
```

{{% /tab %}}
{{< /tabpane >}}

### 지원 모델 목록 불러오기

사용 가능한 모든 모델과 해당 ID를 조회할 수 있습니다. 동적으로 모델을 선택하거나 가능한 모델을 확인할 때 활용하세요.

{{< tabpane text=true >}}
{{% tab header="Bash" value="bash" %}}

```bash
curl https://api.inference.wandb.ai/v1/models \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-api-key>" \
  -H "OpenAI-Project: <your-team>/<your-project>" 
```

{{% /tab %}}
{{% tab header="Python" value="python" %}}

```python
import openai

client = openai.OpenAI(
    base_url="https://api.inference.wandb.ai/v1",
    api_key="<your-api-key>"
)

response = client.models.list()

for model in response.data:
    print(model.id)
```

{{% /tab %}}
{{< /tabpane >}}

## 응답 형식

API는 OpenAI와 호환되는 형식으로 응답을 반환합니다:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Here's a joke for you..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 50,
    "total_tokens": 75
  }
}
```

## 다음 단계

- [사용 예시]({{< relref path="examples" lang="ko" >}})로 API 실행을 바로 체험해보세요
- [UI]({{< relref path="ui-guide" lang="ko" >}})에서 다양한 모델을 살펴보세요
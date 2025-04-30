---
title: OpenAI API
description: OpenAI API와 함께 W&B를 사용하는 방법
menu:
  default:
    identifier: ko-guides-integrations-openai-api
    parent: integrations
weight: 240
---

{{< cta-button colabLink="https://github.com/wandb/examples/blob/master/colabs/openai/OpenAI_API_Autologger_Quickstart.ipynb" >}}

W&B OpenAI API 인테그레이션을 사용하여 모든 OpenAI 모델 (파인튜닝된 모델 포함)에 대한 요청, 응답, 토큰 수 및 모델 메타데이터를 기록합니다.

{{% alert %}}
W&B를 사용하여 파인튜닝 Experiments, Models, Datasets을 추적하고 결과를 동료와 공유하는 방법에 대한 자세한 내용은 [OpenAI 파인튜닝 인테그레이션]({{< relref path="./openai-fine-tuning.md" lang="ko" >}})을 참조하세요.
{{% /alert %}}

API 입력 및 출력을 기록하면 다양한 프롬프트의 성능을 빠르게 평가하고, 다양한 모델 설정 (예: temperature)을 비교하고, 토큰 사용량과 같은 기타 사용량 메트릭을 추적할 수 있습니다.

{{< img src="/images/integrations/open_ai_autolog.png" alt="" >}}

## OpenAI Python API 라이브러리 설치

W&B autolog 인테그레이션은 OpenAI 버전 0.28.1 이하에서 작동합니다.

OpenAI Python API 버전 0.28.1을 설치하려면 다음을 실행합니다.
```python
pip install openai==0.28.1
```

## OpenAI Python API 사용

### 1. autolog 임포트 및 초기화
먼저 `wandb.integration.openai`에서 `autolog`를 임포트하고 초기화합니다.

```python
import os
import openai
from wandb.integration.openai import autolog

autolog({"project": "gpt5"})
```

선택적으로 `wandb.init()`이 허용하는 인수가 있는 사전을 `autolog`에 전달할 수 있습니다. 여기에는 프로젝트 이름, 팀 이름, Entity 등이 포함됩니다. [`wandb.init`]({{< relref path="/ref/python/init.md" lang="ko" >}})에 대한 자세한 내용은 API Reference Guide를 참조하세요.

### 2. OpenAI API 호출
OpenAI API를 호출할 때마다 W&B에 자동으로 기록됩니다.

```python
os.environ["OPENAI_API_KEY"] = "XXX"

chat_request_kwargs = dict(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers"},
        {"role": "user", "content": "Where was it played?"},
    ],
)
response = openai.ChatCompletion.create(**chat_request_kwargs)
```

### 3. OpenAI API 입력 및 응답 보기

**1단계**에서 `autolog`에 의해 생성된 W&B [run]({{< relref path="/guides/models/track/runs/" lang="ko" >}}) 링크를 클릭합니다. 그러면 W&B 앱의 프로젝트 Workspace로 리디렉션됩니다.

생성한 Run을 선택하여 추적 테이블, 추적 타임라인 및 사용된 OpenAI LLM의 모델 아키텍처를 봅니다.

## autolog 끄기
OpenAI API 사용을 마친 후에는 모든 W&B 프로세스를 닫기 위해 `disable()`을 호출하는 것이 좋습니다.

```python
autolog.disable()
```

이제 입력 및 완료가 W&B에 기록되어 분석하거나 동료와 공유할 수 있습니다.

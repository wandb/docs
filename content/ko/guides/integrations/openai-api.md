---
title: OpenAI API
description: W&B를 OpenAI API와 함께 사용하는 방법
menu:
  default:
    identifier: ko-guides-integrations-openai-api
    parent: integrations
weight: 240
---

{{< cta-button colabLink="https://github.com/wandb/examples/blob/master/colabs/openai/OpenAI_API_Autologger_Quickstart.ipynb" >}}

W&B OpenAI API 인테그레이션을 이용하면 모든 OpenAI 모델(파인튜닝된 모델 포함)에 대한 요청, 응답, 토큰 수, 모델 메타데이터를 자동으로 로그할 수 있습니다.

{{% alert %}}
[OpenAI fine-tuning integration]({{< relref path="./openai-fine-tuning.md" lang="ko" >}})에서 W&B를 활용해 fine-tuning 실험, 모델, 데이터셋을 추적하고 동료와 결과를 공유하는 방법을 알아보세요.
{{% /alert %}}

API 입력값과 출력값을 기록하면 다양한 프롬프트의 성능을 빠르게 평가하고, 온도(temperature) 같은 다양한 모델 설정을 비교할 수 있으며, 토큰 사용량 등 기타 메트릭도 편리하게 추적할 수 있습니다.

{{< img src="/images/integrations/open_ai_autolog.png" alt="OpenAI API automatic logging" >}}

## OpenAI Python API 라이브러리 설치

W&B autolog 인테그레이션은 OpenAI 버전 0.28.1 이하에서 사용할 수 있습니다.

OpenAI Python API 0.28.1 버전을 설치하려면 다음 명령어를 실행하세요:
```python
pip install openai==0.28.1
```

## OpenAI Python API 사용법

### 1. autolog 임포트 및 초기화
먼저, `wandb.integration.openai`에서 `autolog`를 임포트한 후 초기화하세요.

```python
import os
import openai
from wandb.integration.openai import autolog

autolog({"project": "gpt5"})
```

`autolog`에는 `wandb.init()`에서 사용할 수 있는 인자를 담은 딕셔너리를 선택적으로 전달할 수 있습니다. 여기에는 프로젝트 이름, 팀 이름, entity 등이 포함됩니다. [`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ko" >}})에 대한 자세한 내용은 API Reference Guide를 참고하세요.

### 2. OpenAI API 호출
이제 OpenAI API를 호출할 때마다 W&B에 자동으로 로그됩니다.

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

### 3. OpenAI API 입력 및 응답 확인하기

**1단계**에서 `autolog`가 생성한 W&B [run]({{< relref path="/guides/models/track/runs/" lang="ko" >}}) 링크를 클릭하세요. 이 링크를 통해 프로젝트 workspace가 열립니다.

해당 run을 선택하면 OpenAI LLM에 대한 trace 테이블, trace 타임라인, 그리고 사용된 모델 아키텍처를 확인할 수 있습니다.

## autolog 끄기
OpenAI API 사용이 끝난 후에는 `disable()`을 호출하여 모든 W&B 프로세스를 종료하는 것이 좋습니다.

```python
autolog.disable()
```

이제 입력값과 결과가 W&B에 저장되어, 분석하거나 동료와 공유할 준비가 완료되었습니다.
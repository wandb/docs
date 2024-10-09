---
title: OpenAI API
description: W&B를 OpenAI API와 함께 사용하는 방법.
slug: /guides/integrations/openai-api
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

Weights & Biases OpenAI API 인테그레이션을 사용하여 모든 OpenAI 모델, 파인튠된 모델을 포함하여 요청, 응답, 토큰 수 및 모델 메타데이터를 1개의 코드 행으로 로그할 수 있습니다.

:::info
W&B autolog 인테그레이션은 `openai <= 0.28.1`와 함께 작동합니다. `pip install openai==0.28.1`로 올바른 버전의 `openai`를 설치하세요.
:::

<CTAButtons colabLink="https://github.com/wandb/examples/blob/master/colabs/openai/OpenAI_API_Autologger_Quickstart.ipynb"></CTAButtons>

이제 단 1개의 코드 행으로 OpenAI Python SDK의 입력과 출력을 Weights & Biases에 자동으로 로그할 수 있습니다!

![](/images/integrations/open_ai_autolog.png)

API 입력과 출력을 로그하기 시작하면 다양한 프롬프트의 성능을 빠르게 평가하고, 온도와 같은 다른 모델 설정을 비교하며, 토큰 사용량과 같은 다른 사용 메트릭을 추적할 수 있습니다.

시작하려면, `wandb` 라이브러리를 pip로 설치한 후 아래 단계를 따르세요:

### 1. autolog 불러오기 및 초기화하기
먼저, `wandb.integration.openai`에서 `autolog`를 불러오고 초기화하세요.

```python
import os
import openai
from wandb.integration.openai import autolog

autolog({"project": "gpt5"})
```

인수 `wandb.init()`가 수용할 수 있는 사전(dictionary)을 `autolog`에 선택적으로 전달할 수 있습니다. 여기에는 프로젝트 이름, 팀 이름, entity 등이 포함됩니다. [`wandb.init`](../../../ref/python/init.md)에 대한 자세한 내용은 API 참조 가이드를 참조하세요.

### 2. OpenAI API 호출하기
OpenAI API에 대한 각 호출은 이제 Weights & Biases에 자동으로 로그됩니다.

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

`autolog`에 의해 생성된 Weights & Biases [run](../../runs/intro.md) 링크를 클릭하여 **1단계**에서 작업 공간으로 리디렉션됩니다.

생성한 run을 선택하여 OpenAI LLM을 사용하여 추적 테이블, 추적 타임라인 및 모델 아키텍처를 확인하세요.

### 4. autolog 비활성화하기
OpenAI API 사용을 마친 후에는 모든 W&B 프로세스를 종료하기 위해 `disable()`을 호출하는 것이 좋습니다.

```python
autolog.disable()
```

이제 입력과 완료 항목이 Weights & Biases에 로그되어 분석되거나 동료와 공유할 준비가 됩니다.
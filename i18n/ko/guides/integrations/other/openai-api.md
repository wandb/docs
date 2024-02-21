---
description: How to use W&B with the OpenAI API.
slug: /guides/integrations/openai-api
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# OpenAI API

Weights & Biases OpenAI API 통합 기능을 사용하여 OpenAI 모델의 요청, 응답, 토큰 수 및 모델 메타데이터를 단 한 줄의 코드로 로그하세요. 파인 튜닝된 모델을 포함한 모든 OpenAI 모델에서 작동합니다.

:::info
W&B autolog 통합은 `openai <= 0.28.1`과 함께 작동합니다. `pip install openai==0.28.1`로 올바른 버전의 `openai`를 설치하세요.
:::

**[여기서 Colab 노트북에서 시도해 보세요 →](https://github.com/wandb/examples/blob/master/colabs/openai/OpenAI_API_Autologger_Quickstart.ipynb)**

단 한 줄의 코드로 이제 OpenAI Python SDK의 입력과 출력을 Weights & Biases에 자동으로 로그할 수 있습니다!

![](/images/integrations/open_ai_autolog.png)

API 입력과 출력을 로깅하기 시작하면 다양한 프롬프트의 성능을 빠르게 평가하고, 다른 모델 설정(예: 온도)을 비교하며, 토큰 사용량과 같은 다른 사용 메트릭을 추적할 수 있습니다.

시작하려면, `wandb` 라이브러리를 pip로 설치한 다음 아래 단계를 따르세요:

### 1. autolog를 가져오고 초기화하기
먼저, `wandb.integration.openai`에서 `autolog`를 가져와서 초기화합니다.

```python
import os
import openai
from wandb.integration.openai import autolog

autolog({"project": "gpt5"})
```

`autolog`에 `wandb.init()`이 받아들이는 인수인 사전을 선택적으로 전달할 수 있습니다. 이에는 프로젝트 이름, 팀 이름, 엔티티 등이 포함됩니다. [`wandb.init`](../../../ref/python/init.md)에 대한 자세한 정보는 API 참조 가이드를 참조하세요.

### 2. OpenAI API 호출하기
이제 OpenAI API에 대한 모든 호출이 자동으로 Weights & Biases에 로그됩니다.

```python
os.environ["OPENAI_API_KEY"] = "XXX"

chat_request_kwargs = dict(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "당신은 도움이 되는 조수입니다."},
        {"role": "user", "content": "2020년에 월드 시리즈에서 우승한 팀은?"},
        {"role": "assistant", "content": "로스앤젤레스 다저스"},
        {"role": "user", "content": "어디서 열렸나요?"},
    ],
)
response = openai.ChatCompletion.create(**chat_request_kwargs)
```

### 3. OpenAI API 입력과 응답 보기

**1단계**에서 `autolog`에 의해 생성된 Weights & Biases [실행](../../runs/intro.md) 링크를 클릭하세요. 이 링크는 W&B 앱의 프로젝트 워크스페이스로 리디렉션됩니다.

생성한 실행을 선택하여 OpenAI LLM의 추적 테이블, 추적 타임라인 및 모델 아키텍처를 볼 수 있습니다.

### 4. autolog 비활성화하기
OpenAI API 사용을 마친 후에는 모든 W&B 프로세스를 닫기 위해 `disable()`을 호출하는 것이 좋습니다.

```python
autolog.disable()
```

이제 입력과 완성이 Weights & Biases에 로깅되어 분석을 위해 준비되거나 동료와 공유할 준비가 되었습니다.
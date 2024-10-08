---
title: DeepChecks
description: W&B와 DeepChecks를 통합하는 방법.
slug: /guides/integrations/deepchecks
displayed_sidebar: default
---
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

<CTAButtons colabLink="https://colab.research.google.com/github/deepchecks/deepchecks/blob/0.5.0-1-g5380093/docs/source/examples/guides/export_outputs_to_wandb.ipynb"></CTAButtons>

DeepChecks는 기계학습 모델과 데이터를 검증하는 데 도움을 주며, 데이터의 무결성을 확인하고, 분포를 검사하며, 데이터 분할을 검증하고, 모델을 평가하며, 다양한 모델 간의 비교를 최소한의 노력으로 수행할 수 있도록 돕습니다.

[DeepChecks와 wandb 통합에 대해 더 읽어보기 ->](https://docs.deepchecks.com/en/stable/examples/guides/export_outputs_to_wandb.html)

## 시작하기

DeepChecks를 Weights & Biases와 함께 사용하려면 먼저 [여기](https://wandb.ai/site)에서 Weights & Biases 계정을 등록해야 합니다. DeepChecks에 있는 Weights & Biases 인테그레이션을 사용하면 다음과 같이 빠르게 시작할 수 있습니다:

```python
import wandb
wandb.login()

# deepchecks에서 당신의 체크를 가져오기
from deepchecks.checks import ModelErrorAnalysis

# 체크를 실행하기
result = ModelErrorAnalysis()...

# 그 결과를 wandb에 전송하기
result.to_wandb()
```

또한 전체 DeepChecks 테스트 스위트를 Weights & Biases에 로그할 수 있습니다.

```python
import wandb
wandb.login()

# deepchecks에서 full_suite 테스트 가져오기
from deepchecks.suites import full_suite

# DeepChecks 테스트 스위트를 생성하고 실행하기
suite_result = full_suite().run(...)

# 결과를 wandb에 전송하기
# 이곳에 필요한 wandb.init 설정과 인수를 전달할 수 있습니다
suite_result.to_wandb(
    project='my-suite-project', 
    config={'suite-name': 'full-suite'}
)
```

## 예제

[**이 Report**](https://wandb.ai/cayush/deepchecks/reports/Validate-your-Data-and-Models-with-Deepchecks-and-W-B--VmlldzoxNjY0ODc5)는 DeepChecks와 Weights & Biases를 사용하는 것의 강력함을 보여줍니다.

![](/images/integrations/deepchecks_example.png)

Weights & Biases 통합에 대한 질문 또는 문제가 있습니까? [DeepChecks GitHub 저장소](https://github.com/deepchecks/deepchecks)에 이슈를 열어주시면, 우리가 잡아서 답변을 드리겠습니다 :)
---
description: How to integrate W&B with DeepChecks.
slug: /guides/integrations/deepchecks
displayed_sidebar: default
---

# DeepChecks

[**여기에서 Colab 노트북으로 시도해보세요 →**](https://colab.research.google.com/github/deepchecks/deepchecks/blob/0.5.0-1-g5380093/docs/source/examples/guides/export\_outputs\_to\_wandb.ipynb)

DeepChecks는 데이터의 무결성을 검증하고, 데이터 분포를 검사하고, 데이터 분할을 검증하며, 모델을 평가하고 서로 다른 모델들을 비교하는 등, 최소한의 노력으로 기계학습 모델과 데이터를 검증하는 데 도움을 줍니다.

[DeepChecks와 Weights & Biases 인테그레이션에 대해 더 읽어보기 ->](https://docs.deepchecks.com/en/stable/examples/guides/export\_outputs\_to\_wandb.html)

## 시작하기

Weights & Biases를 사용하여 DeepChecks를 사용하려면 먼저 [여기](https://wandb.ai/site)에서 Weights & Biases 계정을 등록해야 합니다. DeepChecks에서 Weights & Biases 인테그레이션을 사용하면 다음과 같이 빠르게 시작할 수 있습니다:

```python
import wandb
wandb.login()

# deepchecks에서 검사를 가져옵니다
from deepchecks.checks import ModelErrorAnalysis

# 검사를 실행합니다
result = ModelErrorAnalysis()...

# 그 결과를 wandb에 푸시합니다
result.to_wandb()
```

전체 DeepChecks 테스트 스위트도 Weights & Biases에 로그할 수 있습니다

```python
import wandb
wandb.login()

# deepchecks에서 full_suite 테스트를 가져옵니다
from deepchecks.suites import full_suite

# DeepChecks 테스트 스위트를 생성하고 실행합니다
suite_result = full_suite().run(...)

# 결과를 wandb에 푸시합니다
# 여기서 필요한 모든 wandb.init 설정과 인수를 전달할 수 있습니다
suite_result.to_wandb(
    project='my-suite-project', 
    config={'suite-name': 'full-suite'}
)
```

## 예시

``[**이 리포트**](https://wandb.ai/cayush/deepchecks/reports/Validate-your-Data-and-Models-with-Deepchecks-and-W-B--VmlldzoxNjY0ODc5)는 DeepChecks와 Weights & Biases를 사용하는 것의 강력한 기능을 보여줍니다

![](/images/integrations/deepchecks_example.png)

이 Weights & Biases 인테그레이션에 대해 궁금한 점이나 문제가 있으신가요? [DeepChecks github 저장소](https://github.com/deepchecks/deepchecks)에 이슈를 오픈하시면 답변을 드리기 위해 확인하고 연락드리겠습니다 :)
---
title: DeepChecks
description: DeepChecks와 W&B를 통합하는 방법.
menu:
  default:
    identifier: ko-guides-integrations-deepchecks
    parent: integrations
weight: 60
---

{{< cta-button colabLink="https://colab.research.google.com/github/deepchecks/deepchecks/blob/0.5.0-1-g5380093/docs/source/examples/guides/export_outputs_to_wandb.ipynb" >}}

DeepChecks는 최소한의 노력으로 데이터의 무결성 검증, 분포 검사, 데이터 분할 검증, 모델 평가, 여러 모델 간 비교 등 기계 학습 모델과 데이터를 검증하는 데 도움을 줍니다.

[DeepChecks 및 wandb 통합에 대해 자세히 알아보기 ->](https://docs.deepchecks.com/stable/general/usage/exporting_results/auto_examples/plot_exports_output_to_wandb.html)

## 시작하기

DeepChecks를 Weights & Biases 와 함께 사용하려면 먼저 Weights & Biases 계정을 [여기](https://wandb.ai/site)에서 가입해야 합니다. DeepChecks의 Weights & Biases 인테그레이션을 사용하면 다음과 같이 빠르게 시작할 수 있습니다.

```python
import wandb

wandb.login()

# deepchecks에서 검사 가져오기
from deepchecks.checks import ModelErrorAnalysis

# 검사 실행
result = ModelErrorAnalysis()

# 해당 결과를 wandb로 푸시
result.to_wandb()
```

전체 DeepChecks 테스트 스위트를 Weights & Biases 에 로그할 수도 있습니다.

```python
import wandb

wandb.login()

# deepchecks에서 full_suite 테스트 가져오기
from deepchecks.suites import full_suite

# DeepChecks 테스트 스위트 생성 및 실행
suite_result = full_suite().run(...)

# thes 결과를 wandb로 푸시
# 여기에서 필요한 wandb.init 구성 및 인수를 전달할 수 있습니다.
suite_result.to_wandb(project="my-suite-project", config={"suite-name": "full-suite"})
```

## 예시

``[**이 Report**](https://wandb.ai/cayush/deepchecks/reports/Validate-your-Data-and-Models-with-Deepchecks-and-W-B--VmlldzoxNjY0ODc5)는 DeepChecks와 Weights & Biases를 사용하는 강력한 기능을 보여줍니다.

{{< img src="/images/integrations/deepchecks_example.png" alt="" >}}

이 Weights & Biases 인테그레이션에 대한 질문이나 문제가 있으십니까? [DeepChecks github repository](https://github.com/deepchecks/deepchecks)에 이슈를 열어주시면 확인 후 답변드리겠습니다 :)

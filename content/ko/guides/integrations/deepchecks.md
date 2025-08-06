---
title: 'DeepChecks

  '
description: W&B 를 DeepChecks 와 통합하는 방법
menu:
  default:
    identifier: ko-guides-integrations-deepchecks
    parent: integrations
weight: 60
---

{{< cta-button colabLink="https://colab.research.google.com/github/deepchecks/deepchecks/blob/0.5.0-1-g5380093/docs/source/examples/guides/export_outputs_to_wandb.ipynb" >}}

DeepChecks는 기계학습 모델과 데이터를 검증하는 데 도움을 줍니다. 예를 들어 데이터의 무결성 확인, 분포 분석, 데이터 분할 유효성 검증, 모델 평가 및 여러 모델 간 비교까지 최소한의 노력으로 수행할 수 있습니다.

[DeepChecks와 wandb 인테그레이션에 대해 더 알아보기 ->](https://docs.deepchecks.com/stable/general/usage/exporting_results/auto_examples/plot_exports_output_to_wandb.html)

## 시작하기

DeepChecks를 W&B와 함께 사용하려면 먼저 [W&B 계정](https://wandb.ai/site)에 가입해야 합니다. DeepChecks 내의 W&B 인테그레이션을 통해 아래와 같이 빠르게 시작할 수 있습니다.

```python
import wandb

wandb.login()

# deepchecks에서 check를 불러오기
from deepchecks.checks import ModelErrorAnalysis

# check 실행하기
result = ModelErrorAnalysis()

# 결과를 wandb로 전송
result.to_wandb()
```

DeepChecks의 전체 테스트 스위트도 W&B에 로그로 남길 수 있습니다.

```python
import wandb

wandb.login()

# deepchecks에서 full_suite 테스트를 불러오기
from deepchecks.suites import full_suite

# DeepChecks 테스트 스위트를 생성 및 실행
suite_result = full_suite().run(...)

# 결과를 wandb로 전송
# 이때 필요에 따라 wandb.init 설정과 인수도 전달할 수 있습니다
suite_result.to_wandb(project="my-suite-project", config={"suite-name": "full-suite"})
```

## 예시

[이 Report](https://wandb.ai/cayush/deepchecks/reports/Validate-your-Data-and-Models-with-Deepchecks-and-W-B--VmlldzoxNjY0ODc5)에서는 DeepChecks와 W&B를 함께 사용하는 강력함을 확인할 수 있습니다.

{{< img src="/images/integrations/deepchecks_example.png" alt="Deepchecks 데이터 검증 결과" >}}

이 W&B 인테그레이션에 대해 궁금한 점이나 문제가 있으신가요? [DeepChecks github 저장소](https://github.com/deepchecks/deepchecks)에 이슈를 남겨주시면 신속히 답변드리겠습니다.
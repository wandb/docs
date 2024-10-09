---
title: spaCy
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

[spaCy](https://spacy.io)는 "산업 강도의" NLP 라이브러리로, 빠르고 정확한 모델을 최소한의 번거로움으로 제공합니다. spaCy v3부터 Weights & Biases를 [`spacy train`](https://spacy.io/api/cli#train)과 함께 사용하여 spaCy 모델의 트레이닝 메트릭을 추적하고 모델과 데이터셋을 저장하고 버전 관리할 수 있습니다. 설정 파일에 몇 줄만 추가하면 됩니다!

## 시작하기: 모델 추적 및 저장

### 1. `wandb` 라이브러리 설치 및 로그인

<Tabs
  defaultValue="cli"
  values={[
    {label: 'Command Line', value: 'cli'},
    {label: 'Notebook', value: 'notebook'},
  ]}>
  <TabItem value="cli">

```python
pip install wandb
wandb login
```

  </TabItem>
  <TabItem value="notebook">

```python
!pip install wandb

import wandb
wandb.login()
```

  </TabItem>
</Tabs>

### 2) `WandbLogger`를 spaCy 설정 파일에 추가하기

spaCy 설정 파일은 로그 기록뿐만 아니라 GPU 할당, 옵티마이저 선택, 데이터셋 경로 등 트레이닝의 모든 측면을 지정하는 데 사용됩니다. 최소한 `[training.logger]` 아래에 `@loggers` 키와 `"spacy.WandbLogger.v3"` 값을 제공하고 `project_name`을 추가해야 합니다.

:::info
spaCy 트레이닝 설정 파일의 작동 방식과 트레이닝을 맞춤화하기 위해 전달할 수 있는 다른 옵션에 대한 자세한 내용은 [spaCy의 문서](https://spacy.io/usage/training)를 확인하세요.
:::

```python
[training.logger]
@loggers = "spacy.WandbLogger.v3"
project_name = "my_spacy_project"
remove_config_values = ["paths.train", "paths.dev", "corpora.train.path", "corpora.dev.path"]
log_dataset_dir = "./corpus"
model_log_interval = 1000
```

| Name                   | Description                                                                                                                                                                                                                                                   |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `project_name`         | `str`. Weights & Biases [프로젝트](../app/pages/project-page.md)의 이름. 해당 프로젝트가 존재하지 않으면 자동으로 생성됩니다.                                                                                                    |
| `remove_config_values` | `List[str]` . W&B에 업로드되기 전에 설정에서 제외할 값들의 목록. 기본값은 `[]`.                                                                                                                                                     |
| `model_log_interval`   | `Optional int`. 기본값은 `None`. 설정된 경우, [Artifacts](../artifacts/intro.md)와 함께 [모델 버전 관리](../model_registry/intro.md)가 활성화됩니다. 모델 체크포인트를 로그하는 사이의 단계 수를 전달합니다. 기본값은 `None`. |
| `log_dataset_dir`      | `Optional str`. 경로가 전달되면, 데이터셋이 트레이닝 시작 시 아티팩트로 업로드됩니다. 기본값은 `None`.                                                                                                            |
| `entity`               | `Optional str` . 전달된 경우, 지정된 엔티티에 run이 생성됩니다.                                                                                                                                                                                   |
| `run_name`             | `Optional str` . 지정된 경우, 해당 이름으로 run이 생성됩니다.                                                                                                                                                                               |

### 3) 트레이닝 시작

spaCy 트레이닝 설정에 `WandbLogger`를 추가했다면, 평소와 같이 `spacy train`을 실행할 수 있습니다.

<Tabs
  defaultValue="cli"
  values={[
    {label: 'Command Line', value: 'cli'},
    {label: 'Notebook', value: 'notebook'},
  ]}>
  <TabItem value="cli">

```python
python -m spacy train \
    config.cfg \
    --output ./output \
    --paths.train ./train \
    --paths.dev ./dev
```

  </TabItem>
  <TabItem value="notebook">

```python
!python -m spacy train \
    config.cfg \
    --output ./output \
    --paths.train ./train \
    --paths.dev ./dev
```

  </TabItem>
</Tabs>

트레이닝이 시작되면, 트레이닝 run의 [W&B 페이지](../app/pages/run-page.md) 링크가 출력되어 Weights & Biases 웹 UI에서 실험 추적 [대시보드](../track/app.md)로 이동할 수 있습니다.
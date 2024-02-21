---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# spaCy

[spaCy](https://spacy.io)는 "산업 강도"의 NLP 라이브러리로, 빠르고 정확한 모델을 최소한의 노력으로 제공합니다. spaCy v3부터는 [`spacy train`](https://spacy.io/api/cli#train)과 함께 Weights & Biases를 사용하여 spaCy 모델의 학습 메트릭을 추적하고 모델과 데이터세트를 저장하고 버전 관리할 수 있습니다. 그리고 필요한 것은 구성에 몇 줄 추가하는 것뿐입니다!

## 시작하기: 모델 추적 및 저장

### 1. `wandb` 라이브러리를 설치하고 로그인하기

<Tabs
  defaultValue="cli"
  values={[
    {label: '명령 줄', value: 'cli'},
    {label: '노트북', value: 'notebook'},
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

### 2) spaCy 구성 파일에 `WandbLogger` 추가하기

spaCy 구성 파일은 로깅뿐만 아니라 학습의 모든 측면을 지정하는 데 사용됩니다 -- GPU 할당, 옵티마이저 선택, 데이터세트 경로 등. 최소한, `[training.logger]` 아래에서는 키 `@loggers`에 값 `"spacy.WandbLogger.v3"`와 `project_name`을 제공해야 합니다.

:::info
spaCy 학습 구성 파일 작동 방식과 학습을 사용자 정의하기 위해 전달할 수 있는 다른 옵션에 대해 자세히 알아보려면 [spaCy 문서](https://spacy.io/usage/training)를 확인하세요.
:::

```python
[training.logger]
@loggers = "spacy.WandbLogger.v3"
project_name = "my_spacy_project"
remove_config_values = ["paths.train", "paths.dev", "corpora.train.path", "corpora.dev.path"]
log_dataset_dir = "./corpus"
model_log_interval = 1000
```

| 이름                      | 설명                                                                                                                                                                                                                                                         |
| ---------------------- |--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `project_name`         | `str`. Weights & Biases [프로젝트](../app/pages/project-page.md)의 이름입니다. 프로젝트는 존재하지 않는 경우 자동으로 생성됩니다.                                                                                                                  |
| `remove_config_values` | `List[str]`. W&B에 업로드되기 전에 구성에서 제외할 값 목록입니다. 기본값은 `[]`입니다.                                                                                                                                                                      |
| `model_log_interval`   | `Optional int`. 기본값은 `None`입니다. 설정된 경우, [모델 버전 관리](../model_registry/intro.md)를 [아티팩트](../artifacts/intro.md)와 함께 활성화합니다. 모델 체크포인트를 로깅하는 사이에 기다릴 단계 수를 전달합니다. 기본값은 `None`입니다. |
| `log_dataset_dir`      | `Optional str`. 경로가 전달되면 데이터세트가 학습 시작 시 아티팩트로 업로드됩니다. 기본값은 `None`입니다.                                                                                                                                                    |
| `entity`               | `Optional str`. 전달된 경우, 실행이 지정된 엔티티에서 생성됩니다.                                                                                                                                                                                          |
| `run_name`             | `Optional str`. 지정된 경우, 실행이 지정된 이름으로 생성됩니다.                                                                                                                                                                                             |

### 3) 학습 시작하기

spaCy 학습 구성에 `WandbLogger`를 추가한 후, 평소처럼 `spacy train`을 실행할 수 있습니다.


<Tabs
  defaultValue="cli"
  values={[
    {label: '명령 줄', value: 'cli'},
    {label: '노트북', value: 'notebook'},
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

학습이 시작되면, 학습 실행의 [W&B 페이지](../app/pages/run-page.md)로 가는 링크가 출력되며, 이는 Weights & Biases 웹 UI에서 이 실행의 실험 추적 [대시보드](../track/app.md)로 이동하게 됩니다.
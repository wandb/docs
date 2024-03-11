---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# spaCy

[spaCy](https://spacy.io)는 "산업 강도"의 인기 있는 NLP 라이브러리입니다: 빠르고 정확한 모델을 최소한의 번거로움으로 제공합니다. spaCy v3부터, Weights & Biases는 이제 [`spacy train`](https://spacy.io/api/cli#train)과 함께 사용하여 spaCy 모델의 트레이닝 메트릭을 추적하고 모델 및 데이터셋을 저장하고 버전 관리할 수 있습니다. 그리고 이 모든 것은 설정에 몇 줄만 추가하면 됩니다!

## 시작하기: 모델 추적 및 저장

### 1. `wandb` 라이브러리를 설치하고 로그인

<Tabs
  defaultValue="cli"
  values={[
    {label: '커맨드라인', value: 'cli'},
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

### 2) spaCy 설정 파일에 `WandbLogger` 추가

spaCy 설정 파일은 로그 기록뿐만 아니라 트레이닝의 모든 측면을 지정하는 데 사용됩니다 -- GPU 할당, 옵티마이저 선택, 데이터셋 경로 등. 최소한 `[training.logger]` 아래에서 키 `@loggers`에 값 `"spacy.WandbLogger.v3"`와 `project_name`을 제공해야 합니다.

:::info
spaCy 트레이닝 설정 파일 작동 방식 및 트레이닝을 사용자 정의하기 위해 전달할 수 있는 다른 옵션에 대한 자세한 내용은 [spaCy의 문서](https://spacy.io/usage/training)를 확인하세요.
:::

```python
[training.logger]
@loggers = "spacy.WandbLogger.v3"
project_name = "my_spacy_project"
remove_config_values = ["paths.train", "paths.dev", "corpora.train.path", "corpora.dev.path"]
log_dataset_dir = "./corpus"
model_log_interval = 1000
```

| 이름                   | 설명                                                                                                                                                                                                                                                   |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `project_name`         | `str`. Weights & Biases [프로젝트](../app/pages/project-page.md)의 이름입니다. 프로젝트가 아직 존재하지 않으면 자동으로 생성됩니다.                                                                                                    |
| `remove_config_values` | `List[str]`. W&B에 업로드하기 전에 설정에서 제외할 값들의 목록입니다. 기본값은 `[]`입니다.                                                                                                                                                     |
| `model_log_interval`   | `Optional int`. 기본값은 `None`입니다. 설정하면 [모델 버전 관리](../model_registry/intro.md)가 [아티팩트](../artifacts/intro.md)와 함께 활성화됩니다. 모델 체크포인트를 로그하는 사이의 스텝 수를 전달합니다. 기본값은 `None`입니다. |
| `log_dataset_dir`      | `Optional str`. 경로가 전달되면 데이터셋이 트레이닝 시작 시 아티팩트로 업로드됩니다. 기본값은 `None`입니다.                                                                                                            |
| `entity`               | `Optional str`. 전달되면 run이 지정된 엔티티에서 생성됩니다.                                                                                                                                                                                   |
| `run_name`             | `Optional str`. 지정되면 run이 지정된 이름으로 생성됩니다.                                                                                                                                                                               |

### 3) 트레이닝 시작

spaCy 트레이닝 설정에 `WandbLogger`를 추가한 후에는 평소와 같이 `spacy train`을 실행할 수 있습니다.


<Tabs
  defaultValue="cli"
  values={[
    {label: '커맨드라인', value: 'cli'},
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

트레이닝이 시작되면, 트레이닝 run의 [W&B 페이지](../app/pages/run-page.md) 링크가 출력되며, 여기를 클릭하면 Weights & Biases 웹 UI에서 이 run의 실험 추적 [대시보드](../track/app.md)로 이동할 수 있습니다.
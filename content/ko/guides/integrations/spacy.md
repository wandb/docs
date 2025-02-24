---
title: spaCy
menu:
  default:
    identifier: ko-guides-integrations-spacy
    parent: integrations
weight: 410
---

[spaCy](https://spacy.io)는 빠르고 정확한 모델을 최소한의 번거로움으로 제공하는 인기 있는 "산업용" NLP 라이브러리입니다. spaCy v3부터 Weights & Biases를 [`spacy train`](https://spacy.io/api/cli#train)과 함께 사용하여 spaCy 모델의 트레이닝 메트릭을 추적하고 모델과 데이터셋을 저장하고 버전을 관리할 수 있습니다. 설정에 몇 줄만 추가하면 됩니다.

## 가입하고 API 키를 생성하세요

API 키는 사용자의 장치를 W&B에 인증합니다. 사용자 프로필에서 API 키를 생성할 수 있습니다.

{{% alert %}}
보다 간소화된 접근 방식을 위해 [https://wandb.ai/authorize](https://wandb.ai/authorize)로 직접 이동하여 API 키를 생성할 수 있습니다. 표시된 API 키를 복사하여 비밀번호 관리자와 같은 안전한 위치에 저장하세요.
{{% /alert %}}

1. 오른쪽 상단 모서리에 있는 사용자 프로필 아이콘을 클릭합니다.
2. **User Settings**를 선택한 다음 **API Keys** 섹션으로 스크롤합니다.
3. **Reveal**을 클릭합니다. 표시된 API 키를 복사합니다. API 키를 숨기려면 페이지를 새로 고침하세요.

## `wandb` 라이브러리를 설치하고 로그인하세요

`wandb` 라이브러리를 로컬에 설치하고 로그인하려면 다음을 수행하세요.

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. API 키로 `WANDB_API_KEY` [환경 변수]({{< relref path="/guides/models/track/environment-variables.md" lang="ko" >}})를 설정합니다.

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

2. `wandb` 라이브러리를 설치하고 로그인합니다.

    ```shell
    pip install wandb

    wandb login
    ```

{{% /tab %}}

{{% tab header="Python" value="python" %}}

```bash
pip install wandb
```
```python
import wandb
wandb.login()
```

{{% /tab %}}

{{% tab header="Python notebook" value="notebook" %}}

```notebook
!pip install wandb

import wandb
wandb.login()
```

{{% /tab %}}
{{< /tabpane >}}

## spaCy 설정 파일에 `WandbLogger`를 추가하세요

spaCy 설정 파일은 로깅뿐만 아니라 트레이닝의 모든 측면(GPU 할당, 옵티마이저 선택, 데이터셋 경로 등)을 지정하는 데 사용됩니다. 최소한 `[training.logger]` 아래에 `@loggers` 키와 `"spacy.WandbLogger.v3"` 값을 `project_name`과 함께 제공해야 합니다.

{{% alert %}}
spaCy 트레이닝 설정 파일의 작동 방식과 트레이닝을 사용자 정의하기 위해 전달할 수 있는 기타 옵션에 대한 자세한 내용은 [spaCy 설명서](https://spacy.io/usage/training)를 확인하세요.
{{% /alert %}}

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
| `project_name`         | `str`. W&B Project의 이름입니다. 아직 존재하지 않으면 프로젝트가 자동으로 생성됩니다.                                                                                                                                                                                     |
| `remove_config_values` | `List[str]`. W&B에 업로드하기 전에 설정에서 제외할 값 목록입니다. 기본적으로 `[]`입니다.                                                                                                                                                                                        |
| `model_log_interval`   | `Optional int`. 기본적으로 `None`입니다. 설정하면 [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ko" >}})를 사용한 [모델 버전 관리]({{< relref path="/guides/models/registry/model_registry/" lang="ko" >}})이 활성화됩니다. 모델 체크포인트 로깅 사이의 단계 수를 전달합니다. 기본적으로 `None`입니다. |
| `log_dataset_dir`      | `Optional str`. 경로가 전달되면 트레이닝 시작 시 데이터셋이 Artifact로 업로드됩니다. 기본적으로 `None`입니다.                                                                                                                                                           |
| `entity`               | `Optional str`. 전달되면 지정된 엔티티에 run이 생성됩니다.                                                                                                                                                                                                                 |
| `run_name`             | `Optional str`. 지정되면 지정된 이름으로 run이 생성됩니다.                                                                                                                                                                                                                 |

## 트레이닝 시작

spaCy 트레이닝 설정에 `WandbLogger`를 추가했으면 평소와 같이 `spacy train`을 실행할 수 있습니다.

{{< tabpane text=true >}}

{{% tab header="Command Line" value="cli" %}}

```python
python -m spacy train \
    config.cfg \
    --output ./output \
    --paths.train ./train \
    --paths.dev ./dev
```

{{% /tab %}}

{{% tab header="Python" value="python" %}}

```python
python -m spacy train \
    config.cfg \
    --output ./output \
    --paths.train ./train \
    --paths.dev ./dev
```

{{% /tab %}}

{{% tab header="Python notebook" value="notebook" %}}

```notebook
!python -m spacy train \
    config.cfg \
    --output ./output \
    --paths.train ./train \
    --paths.dev ./dev
```

{{% /tab %}}
{{< /tabpane >}}

트레이닝이 시작되면 트레이닝 run의 [W&B 페이지]({{< relref path="/guides/models/track/runs/" lang="ko" >}}) 링크가 출력됩니다. 이 링크를 클릭하면 Weights & Biases 웹 UI에서 이 run의 실험 추적 [대시보드]({{< relref path="/guides/models/track/workspaces.md" lang="ko" >}})로 이동합니다.

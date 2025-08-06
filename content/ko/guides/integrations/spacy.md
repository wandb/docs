---
title: spaCy
menu:
  default:
    identifier: ko-guides-integrations-spacy
    parent: integrations
weight: 410
---

[spaCy](https://spacy.io)는 "산업용" NLP 라이브러리로 유명합니다. 빠르고 정확한 모델을 최소한의 설정으로 사용할 수 있습니다. spaCy v3부터는 [`spacy train`](https://spacy.io/api/cli#train)에서 W&B를 연동하여 spaCy 모델의 트레이닝 메트릭을 추적하고, Models 및 Datasets를 저장하고 버전 관리할 수 있습니다. 단 몇 줄의 설정만 추가하면 됩니다.

## 회원가입 및 API 키 생성하기

API 키는 W&B에 머신을 인증하는 역할을 합니다. API 키는 사용자 프로필에서 생성할 수 있습니다.

{{% alert %}}
더 간편하게 진행하려면 [W&B 인증 페이지](https://wandb.ai/authorize)에 접속하여 API 키를 생성할 수 있습니다. 표시되는 API 키를 복사해서 비밀번호 관리자와 같은 안전한 위치에 저장해두세요.
{{% /alert %}}

1. 오른쪽 상단의 사용자 프로필 아이콘을 클릭합니다.
1. **User Settings**로 이동한 후 **API Keys** 섹션까지 스크롤합니다.
1. **Reveal**을 클릭해 표시되는 API 키를 복사합니다. 키를 숨기려면 페이지를 새로고침하세요.

## `wandb` 라이브러리 설치 및 로그인

로컬 환경에 `wandb` 라이브러리를 설치하고 로그인하려면:

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [환경 변수]({{< relref path="/guides/models/track/environment-variables.md" lang="ko" >}})에 본인의 API 키를 입력합니다.

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

1. `wandb` 라이브러리를 설치하고 로그인합니다.

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

## spaCy config 파일에 `WandbLogger` 추가하기

spaCy config 파일은 로그 설정뿐만 아니라 트레이닝 전체를 제어합니다. 예를 들어 GPU 할당, 옵티마이저 선택, 데이터셋 경로 등도 지정합니다. 최소한 `[training.logger]` 아래에 `@loggers` 키에 `"spacy.WandbLogger.v3"` 값을 입력하고, `project_name`을 명시하면 됩니다.

{{% alert %}}
spaCy의 트레이닝 config 파일 구조와 트레이닝을 커스터마이즈할 수 있는 옵션에 대한 자세한 내용은 [spaCy 공식 문서](https://spacy.io/usage/training)를 참고하세요.
{{% /alert %}}

```python
[training.logger]
@loggers = "spacy.WandbLogger.v3"
project_name = "my_spacy_project"
remove_config_values = ["paths.train", "paths.dev", "corpora.train.path", "corpora.dev.path"]
log_dataset_dir = "./corpus"
model_log_interval = 1000
```

| 이름                    | 설명                                                                                                                                                                                                                                                    |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `project_name`         | `str`. W&B Project 이름입니다. 프로젝트가 없을 경우 자동으로 생성됩니다.                                                                                                   |
| `remove_config_values` | `List[str]`. W&B에 업로드하기 전 config 파일에서 제외할 값의 리스트입니다. 기본값은 `[]`입니다.                                                                                                                         |
| `model_log_interval`   | `Optional int`. 기본값은 `None`입니다. 지정하면 [model 버전 관리]({{< relref path="/guides/core/registry/" lang="ko" >}})가 [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ko" >}})로 활성화됩니다. 모델 체크포인트를 저장할 스텝 간격을 입력할 수 있습니다. 기본값은 `None`입니다. |
| `log_dataset_dir`      | `Optional str`. 경로를 입력하면 트레이닝 시작 시 데이터셋이 Artifact로 업로드됩니다. 기본값은 `None`입니다.                                                                                                         |
| `entity`               | `Optional str`. 입력할 경우 해당 entity에 run이 생성됩니다.                                                                                                                                                          |
| `run_name`             | `Optional str`. 지정하면 해당 이름으로 run이 생성됩니다.                                                                                                                                                             |

## 트레이닝 시작하기

spaCy 트레이닝 config에 `WandbLogger`를 추가하면 기존과 동일하게 `spacy train` 명령어로 트레이닝을 시작할 수 있습니다.

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

트레이닝이 시작되면, 해당 트레이닝 run의 [W&B 페이지]({{< relref path="/guides/models/track/runs/" lang="ko" >}})로 연결되는 링크가 출력됩니다. 이를 통해 웹 UI에서 해당 실험의 트래킹 [대시보드]({{< relref path="/guides/models/track/workspaces.md" lang="ko" >}})로 바로 이동할 수 있습니다.
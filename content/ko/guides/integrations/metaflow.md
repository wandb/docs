---
title: Metaflow
description: Metaflow와 W&B를 통합하는 방법
menu:
  default:
    identifier: ko-guides-integrations-metaflow
    parent: integrations
weight: 200
---

## 개요

[Metaflow](https://docs.metaflow.org)는 ML 워크플로우를 생성하고 실행하기 위해 [Netflix](https://netflixtechblog.com)에서 만든 프레임워크입니다.

이 인테그레이션을 통해 사용자는 Metaflow [단계 및 흐름](https://docs.metaflow.org/metaflow/basics)에 데코레이터를 적용하여 파라미터와 Artifacts를 W&B에 자동으로 기록할 수 있습니다.

* 단계를 데코레이팅하면 해당 단계 내의 특정 유형에 대한 로깅이 켜지거나 꺼집니다.
* 흐름을 데코레이팅하면 흐름의 모든 단계에 대한 로깅이 켜지거나 꺼집니다.

## 퀵스타트

### 가입 및 API 키 생성

API 키는 사용자의 장치를 W&B에 인증합니다. 사용자 프로필에서 API 키를 생성할 수 있습니다.

{{% alert %}}
보다 간소화된 방법으로 [https://wandb.ai/authorize](https://wandb.ai/authorize)로 직접 이동하여 API 키를 생성할 수 있습니다. 표시된 API 키를 복사하여 비밀번호 관리자와 같은 안전한 위치에 저장하십시오.
{{% /alert %}}

1. 오른쪽 상단 모서리에 있는 사용자 프로필 아이콘을 클릭합니다.
2. **User Settings**를 선택한 다음 **API Keys** 섹션으로 스크롤합니다.
3. **Reveal**을 클릭합니다. 표시된 API 키를 복사합니다. API 키를 숨기려면 페이지를 새로 고칩니다.

### `wandb` 라이브러리 설치 및 로그인

`wandb` 라이브러리를 로컬에 설치하고 로그인하려면:

{{< tabpane text=true >}}
{{% tab header="커맨드라인" value="cli" %}}

1. `WANDB_API_KEY` [환경 변수]({{< relref path="/guides/models/track/environment-variables.md" lang="ko" >}})를 API 키로 설정합니다.

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

2. `wandb` 라이브러리를 설치하고 로그인합니다.

    ```shell
    pip install -Uqqq metaflow fastcore wandb

    wandb login
    ```

{{% /tab %}}

{{% tab header="Python" value="python" %}}

```bash
pip install -Uqqq metaflow fastcore wandb
```
```python
import wandb
wandb.login()
```

{{% /tab %}}

{{% tab header="Python 노트북" value="notebook" %}}

```notebook
!pip install -Uqqq metaflow fastcore wandb

import wandb
wandb.login()
```

{{% /tab %}}
{{< /tabpane >}}

### 흐름 및 단계 데코레이팅

{{< tabpane text=true >}}
{{% tab header="단계" value="step" %}}

단계를 데코레이팅하면 해당 단계 내의 특정 유형에 대한 로깅이 켜지거나 꺼집니다.

이 예에서는 `start`의 모든 데이터셋과 Models가 기록됩니다.

```python
from wandb.integration.metaflow import wandb_log

class WandbExampleFlow(FlowSpec):
    @wandb_log(datasets=True, models=True, settings=wandb.Settings(...))
    @step
    def start(self):
        self.raw_df = pd.read_csv(...).    # pd.DataFrame -> upload as dataset
        self.model_file = torch.load(...)  # nn.Module    -> upload as model
        self.next(self.transform)
```
{{% /tab %}}

{{% tab header="흐름" value="flow" %}}

흐름을 데코레이팅하는 것은 구성 단계를 모두 기본값으로 데코레이팅하는 것과 같습니다.

이 경우 `WandbExampleFlow`의 모든 단계는 기본적으로 데이터셋과 Models를 기록하도록 기본 설정되어 있습니다. 이는 각 단계를 `@wandb_log(datasets=True, models=True)`로 데코레이팅하는 것과 같습니다.

```python
from wandb.integration.metaflow import wandb_log

@wandb_log(datasets=True, models=True)  # decorate all @step 
class WandbExampleFlow(FlowSpec):
    @step
    def start(self):
        self.raw_df = pd.read_csv(...).    # pd.DataFrame -> upload as dataset
        self.model_file = torch.load(...)  # nn.Module    -> upload as model
        self.next(self.transform)
```
{{% /tab %}}

{{% tab header="흐름 및 단계" value="flow_and_steps" %}}

흐름을 데코레이팅하는 것은 모든 단계를 기본값으로 데코레이팅하는 것과 같습니다. 즉, 나중에 다른 `@wandb_log`로 단계를 데코레이팅하면 흐름 수준 데코레이션이 재정의됩니다.

이 예에서:

* `start` 및 `mid`는 데이터셋과 Models를 모두 기록합니다.
* `end`는 데이터셋과 Models를 모두 기록하지 않습니다.

```python
from wandb.integration.metaflow import wandb_log

@wandb_log(datasets=True, models=True)  # same as decorating start and mid
class WandbExampleFlow(FlowSpec):
  # this step will log datasets and models
  @step
  def start(self):
    self.raw_df = pd.read_csv(...).    # pd.DataFrame -> upload as dataset
    self.model_file = torch.load(...)  # nn.Module    -> upload as model
    self.next(self.mid)

  # this step will also log datasets and models
  @step
  def mid(self):
    self.raw_df = pd.read_csv(...).    # pd.DataFrame -> upload as dataset
    self.model_file = torch.load(...)  # nn.Module    -> upload as model
    self.next(self.end)

  # this step is overwritten and will NOT log datasets OR models
  @wandb_log(datasets=False, models=False)
  @step
  def end(self):
    self.raw_df = pd.read_csv(...).    
    self.model_file = torch.load(...)
```
{{% /tab %}}
{{< /tabpane >}}

## 프로그램 방식으로 데이터에 액세스

[`wandb` 클라이언트 라이브러리]({{< relref path="/ref/python/" lang="ko" >}})를 사용하여 기록 중인 원래 Python 프로세스 내부, [웹 앱 UI]({{< relref path="/guides/models/track/workspaces.md" lang="ko" >}}) 또는 [Public API]({{< relref path="/ref/python/public-api/" lang="ko" >}})를 사용하여 프로그램 방식으로 캡처한 정보에 액세스할 수 있습니다. `Parameter`는 W&B의 [`config`]({{< relref path="/guides/models/track/config.md" lang="ko" >}})에 저장되며 [Overview 탭]({{< relref path="/guides/models/track/runs/#overview-tab" lang="ko" >}})에서 찾을 수 있습니다. `datasets`, `models` 및 `others`는 [W&B Artifacts]({{< relref path="/guides/core/artifacts/" lang="ko" >}})에 저장되며 [Artifacts 탭]({{< relref path="/guides/models/track/runs/#artifacts-tab" lang="ko" >}})에서 찾을 수 있습니다. 기본 Python 유형은 W&B의 [`summary`]({{< relref path="/guides/models/track/log/" lang="ko" >}}) 사전에 저장되며 Overview 탭에서 찾을 수 있습니다. API를 사용하여 외부에서 이 정보를 프로그램 방식으로 가져오는 방법에 대한 자세한 내용은 [Public API 가이드]({{< relref path="/guides/models/track/public-api-guide.md" lang="ko" >}})를 참조하십시오.

### 빠른 참조

| 데이터                                            | 클라이언트 라이브러리                            | UI                    |
| ----------------------------------------------- | ----------------------------------------- | --------------------- |
| `Parameter(...)`                                | `wandb.config`                            | Overview 탭, Config  |
| `datasets`, `models`, `others`                  | `wandb.use_artifact("{var_name}:latest")` | Artifacts 탭         |
| 기본 Python 유형 (`dict`, `list`, `str`, etc.) | `wandb.summary`                           | Overview 탭, Summary |

### `wandb_log` kwargs

| kwarg      | 옵션                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `datasets` | <ul><li><code>True</code>: 데이터셋인 인스턴스 변수 기록</li><li><code>False</code></li></ul>                                                                                                                                                                                                                                                                                                                                                                         |
| `models`   | <ul><li><code>True</code>: 모델인 인스턴스 변수 기록</li><li><code>False</code></li></ul>                                                                                                                                                                                                                                                                                                                                                                           |
| `others`   | <ul><li><code>True</code>: 직렬화 가능한 다른 모든 항목을 피클로 기록</li><li><code>False</code></li></ul>                                                                                                                                                                                                                                                                                                                                                                |
| `settings` | <ul><li><code>wandb.Settings(...)</code>: 이 단계 또는 흐름에 대한 사용자 지정 <code>wandb</code> 설정을 지정합니다.</li><li><code>None</code>: <code>wandb.Settings()</code>를 전달하는 것과 같습니다.</li></ul><p>기본적으로:</p><ul><li><code>settings.run_group</code>이 <code>None</code>이면 <code>\{flow_name\}/\{run_id\}</code>로 설정됩니다.</li><li><code>settings.run_job_type</code>이 <code>None</code>이면 <code>\{run_job_type\}/\{step_name\}</code>으로 설정됩니다.</li></ul> |

## 자주 묻는 질문

### 정확히 무엇을 기록합니까? 모든 인스턴스 및 로컬 변수를 기록합니까?

`wandb_log`는 인스턴스 변수만 기록합니다. 로컬 변수는 절대 기록되지 않습니다. 이는 불필요한 데이터 로깅을 피하는 데 유용합니다.

### 어떤 데이터 유형이 기록됩니까?

현재 다음과 같은 유형을 지원합니다.

| 로깅 설정     | 유형                                                                                                                        |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| 기본값 (항상 켜짐) | <ul><li><code>dict, list, set, str, int, float, bool</code></li></ul>                                                       |
| `datasets`          | <ul><li><code>pd.DataFrame</code></li><li><code>pathlib.Path</code></li></ul>                                               |
| `models`            | <ul><li><code>nn.Module</code></li><li><code>sklearn.base.BaseEstimator</code></li></ul>                                    |
| `others`            | <ul><li><a href="https://wiki.python.org/moin/UsingPickle">피클 가능</a>하고 JSON 직렬화 가능한 모든 항목</li></ul> |

### 로깅 행동을 어떻게 구성할 수 있습니까?

| 변수 종류 | 행동                          | 예          | 데이터 유형      |
| ---------------- | ------------------------------ | --------------- | -------------- |
| 인스턴스         | 자동 기록                     | `self.accuracy` | `float`        |
| 인스턴스         | `datasets=True`인 경우 기록     | `self.df`       | `pd.DataFrame` |
| 인스턴스         | `datasets=False`인 경우 기록 안 함 | `self.df`       | `pd.DataFrame` |
| 로컬            | 절대 기록 안 함                | `accuracy`      | `float`        |
| 로컬            | 절대 기록 안 함                | `df`            | `pd.DataFrame` |

### Artifact 계보가 추적됩니까?

예. Artifact가 A 단계의 출력이고 B 단계의 입력인 경우 계보 DAG가 자동으로 구성됩니다.

이 행동의 예는 이 [노트북](https://colab.research.google.com/drive/1wZG-jYzPelk8Rs2gIM3a71uEoG46u_nG#scrollTo=DQQVaKS0TmDU)과 해당 [W&B Artifacts 페이지](https://wandb.ai/megatruong/metaflow_integration/artifacts/dataset/raw_df/7d14e6578d3f1cfc72fe/graph)를 참조하십시오.

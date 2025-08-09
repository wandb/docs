---
title: Metaflow
description: W&B 를 Metaflow 와 통합하는 방법
menu:
  default:
    identifier: ko-guides-integrations-metaflow
    parent: integrations
weight: 200
---

## 개요

[Metaflow](https://docs.metaflow.org)는 [Netflix](https://netflixtechblog.com)에서 만든 ML 워크플로우를 생성하고 실행하기 위한 프레임워크입니다.

이 인테그레이션을 통해 사용자는 Metaflow의 [steps and flows](https://docs.metaflow.org/metaflow/basics)에 데코레이터를 적용하여 파라미터와 Artifacts를 자동으로 W&B에 로그할 수 있습니다.

* step에 데코레이터를 적용하면 해당 step 내 특정 타입에 대한 로깅을 켜거나 끌 수 있습니다.
* flow에 데코레이터를 적용하면 flow 내 모든 step에 대해 로깅을 켜거나 끌 수 있습니다.

## 퀵스타트

### 회원가입 및 API 키 생성

API 키는 W&B에서 머신의 인증에 사용됩니다. API 키는 사용자 프로필에서 생성할 수 있습니다.

{{% alert %}}
더 간편한 방법으로 [W&B 인증 페이지](https://wandb.ai/authorize)에서 바로 API 키를 생성할 수 있습니다. 표시되는 API 키를 복사해서 암호 관리 프로그램 등 안전한 곳에 보관하세요.
{{% /alert %}}

1. 오른쪽 상단에서 사용자 프로필 아이콘을 클릭하세요.
1. **User Settings**를 선택한 뒤 **API Keys** 섹션까지 스크롤하세요.
1. **Reveal**을 클릭하고, 표시되는 API 키를 복사하세요. API 키를 다시 가리려면 페이지를 새로고침하세요.

### `wandb` 라이브러리 설치 및 로그인

로컬에 `wandb` 라이브러리를 설치하고 로그인하려면:

{{% alert %}}
`wandb` 버전이 0.19.8 이하라면, `plum-dispatch` 대신 `fastcore` 1.8.0 이하(`fastcore<1.8.0`)를 설치해야 합니다.
{{% /alert %}}

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [환경 변수]({{< relref path="/guides/models/track/environment-variables.md" lang="ko" >}})에 본인의 API 키를 설정하세요.

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

1. `wandb` 라이브러리를 설치하고 로그인합니다.

    ```shell
    pip install -Uqqq metaflow "plum-dispatch<3.0.0" wandb

    wandb login
    ```

{{% /tab %}}

{{% tab header="Python" value="python" %}}

```bash
pip install -Uqqq metaflow "plum-dispatch<3.0.0" wandb
```
```python
import wandb
wandb.login()
```

{{% /tab %}}

{{% tab header="Python notebook" value="notebook" %}}

```notebook
!pip install -Uqqq metaflow "plum-dispatch<3.0.0" wandb

import wandb
wandb.login()
```

{{% /tab %}}
{{< /tabpane >}}

### flows와 steps 데코레이트하기

{{< tabpane text=true >}}
{{% tab header="Step" value="step" %}}

step에 데코레이터를 적용하면 해당 step 내 특정 타입의 로깅을 켜거나 끌 수 있습니다.

아래 예시에서 `start` 내의 모든 datasets와 models가 로그됩니다.

```python
from wandb.integration.metaflow import wandb_log

class WandbExampleFlow(FlowSpec):
    @wandb_log(datasets=True, models=True, settings=wandb.Settings(...))
    @step
    def start(self):
        self.raw_df = pd.read_csv(...).    # pd.DataFrame -> 데이터셋으로 업로드
        self.model_file = torch.load(...)  # nn.Module    -> 모델로 업로드
        self.next(self.transform)
```
{{% /tab %}}

{{% tab header="Flow" value="flow" %}}

flow에 데코레이터를 적용하면 모든 포함된 step들을 기본값으로 동일하게 데코레이트하는 것과 같습니다.

이 경우, `WandbExampleFlow` 내의 모든 step이 datasets와 models를 기본적으로 로그하며, 이는 각각의 step에 `@wandb_log(datasets=True, models=True)`를 다는 효과와 같습니다.

```python
from wandb.integration.metaflow import wandb_log

@wandb_log(datasets=True, models=True)  # 모든 @step에 데코레이트
class WandbExampleFlow(FlowSpec):
    @step
    def start(self):
        self.raw_df = pd.read_csv(...).    # pd.DataFrame -> 데이터셋으로 업로드
        self.model_file = torch.load(...)  # nn.Module    -> 모델로 업로드
        self.next(self.transform)
```
{{% /tab %}}

{{% tab header="Flow and Steps" value="flow_and_steps" %}}

flow에 데코레이트하면 모든 step을 기본값으로 데코레이트한 것과 같습니다. 나중에 특정 Step에 다른 `@wandb_log`를 추가하면, 해당 step에서는 flow 레벨의 데코레이터가 덮어씌워집니다.

예를 들면:

* `start`와 `mid`는 datasets와 models 모두 로그합니다.
* `end`는 datasets나 models를 모두 로그하지 않습니다.

```python
from wandb.integration.metaflow import wandb_log

@wandb_log(datasets=True, models=True)  # start와 mid에 데코레이트하는 것과 동일
class WandbExampleFlow(FlowSpec):
  # 이 step은 datasets와 models를 로그합니다.
  @step
  def start(self):
    self.raw_df = pd.read_csv(...).    # pd.DataFrame -> 데이터셋으로 업로드
    self.model_file = torch.load(...)  # nn.Module    -> 모델로 업로드
    self.next(self.mid)

  # 이 step도 datasets와 models를 로그합니다.
  @step
  def mid(self):
    self.raw_df = pd.read_csv(...).    # pd.DataFrame -> 데이터셋으로 업로드
    self.model_file = torch.load(...)  # nn.Module    -> 모델로 업로드
    self.next(self.end)

  # 이 step은 오버라이드되어 datasets나 models를 로그하지 않습니다.
  @wandb_log(datasets=False, models=False)
  @step
  def end(self):
    self.raw_df = pd.read_csv(...).    
    self.model_file = torch.load(...)
```
{{% /tab %}}
{{< /tabpane >}}

## 데이터 프로그래밍 방식으로 엑세스하기

기록된 정보를 프로그래밍 방식으로 가져오는 방법은 세 가지가 있습니다: 로그를 찍은 파이썬 프로세스 내부에서 [`wandb` 클라이언트 라이브러리]({{< relref path="/ref/python/" lang="ko" >}}) 사용, [웹 앱 UI]({{< relref path="/guides/models/track/workspaces.md" lang="ko" >}}), 또는 [Public API]({{< relref path="/ref/python/public-api/" lang="ko" >}})를 사용한 프로그래밍적 접근이 있습니다. `Parameter`는 W&B의 [`config`]({{< relref path="/guides/models/track/config.md" lang="ko" >}})에 저장되고, [Overview 탭]({{< relref path="/guides/models/track/runs/#overview-tab" lang="ko" >}})에서 확인할 수 있습니다. `datasets`, `models`, `others`는 [W&B Artifacts]({{< relref path="/guides/core/artifacts/" lang="ko" >}})에 저장되어 [Artifacts 탭]({{< relref path="/guides/models/track/runs/#artifacts-tab" lang="ko" >}})에서 볼 수 있습니다. 기본 파이썬 타입은 W&B의 [`summary`]({{< relref path="/guides/models/track/log/" lang="ko" >}}) dict에 저장되며 Overview 탭에서 확인 가능합니다. Public API를 통한 정보 획득에 자세한 내용은 [Public API 가이드]({{< relref path="/guides/models/track/public-api-guide.md" lang="ko" >}})를 참고하세요.

### 빠른 참고

| 데이터                                         | 클라이언트 라이브러리                                  | UI                       |
| ----------------------------------------------- | ------------------------------------------------------ | ------------------------ |
| `Parameter(...)`                               | `wandb.Run.config`                                     | Overview 탭, Config      |
| `datasets`, `models`, `others`                 | `wandb.Run.use_artifact("{var_name}:latest")`          | Artifacts 탭             |
| 파이썬 기본 타입 (`dict`, `list`, `str`, 등)    | `wandb.Run.summary`                                    | Overview 탭, Summary     |

### `wandb_log` kwargs

| kwarg       | 옵션                                                                                                   |
| ----------- | ------------------------------------------------------------------------------------------------------ |
| `datasets`  | <ul><li><code>True</code>: 인스턴스 변수 중 데이터셋인 것 로그</li><li><code>False</code></li></ul>    |
| `models`    | <ul><li><code>True</code>: 인스턴스 변수 중 모델인 것 로그</li><li><code>False</code></li></ul>        |
| `others`    | <ul><li><code>True</code>: 기타 pickle로 직렬화될 수 있는 것 로그</li><li><code>False</code></li></ul> |
| `settings`  | <ul><li><code>wandb.Settings(...)</code>: 해당 step이나 flow에 사용할 <code>wandb</code> 설정 지정</li><li><code>None</code>: <code>wandb.Settings()</code>와 동일</li></ul><p>기본적으로:</p><ul><li><code>settings.run_group</code>이 <code>None</code>이면 <code>\{flow_name\}/\{run_id\}</code>로,</li><li><code>settings.run_job_type</code>이 <code>None</code>이면 <code>\{run_job_type\}/\{step_name\}</code>로 세팅됩니다.</li></ul> |

## 자주 묻는 질문

### 정확히 무엇을 로그하나요? 모든 인스턴스 및 로컬 변수를 로그하나요?

`wandb_log`는 인스턴스 변수만 로그합니다. 로컬 변수는 절대 로그되지 않습니다. 이는 불필요한 데이터 로그를 피하는 데 유용합니다.

### 어떤 데이터 타입이 로그되나요?

현재 지원하는 타입은 다음과 같습니다:

| 로깅 설정            | 타입                                                                                                 |
| ------------------- | ---------------------------------------------------------------------------------------------------- |
| 기본 (항상 활성화)  | <ul><li><code>dict, list, set, str, int, float, bool</code></li></ul>                                 |
| `datasets`          | <ul><li><code>pd.DataFrame</code></li><li><code>pathlib.Path</code></li></ul>                         |
| `models`            | <ul><li><code>nn.Module</code></li><li><code>sklearn.base.BaseEstimator</code></li></ul>              |
| `others`            | <ul><li><a href="https://wiki.python.org/moin/UsingPickle">pickle-able</a> 하고 JSON 직렬화 가능한 모든 값</li></ul> |

### 로깅 행동은 어떻게 설정하나요?

| 변수 종류     | 행동                                   | 예시              | 데이터 타입        |
| ------------- | -------------------------------------- | ----------------- | ----------------- |
| 인스턴스      | 자동 로그                              | `self.accuracy`   | `float`           |
| 인스턴스      | `datasets=True`일 때 로그              | `self.df`         | `pd.DataFrame`    |
| 인스턴스      | `datasets=False`면 로그하지 않음        | `self.df`         | `pd.DataFrame`    |
| 로컬          | 절대 로그하지 않음                     | `accuracy`        | `float`           |
| 로컬          | 절대 로그하지 않음                     | `df`              | `pd.DataFrame`    |

### artifact 계보(lineage)가 추적되나요?

네. 만약 하나의 artifact가 step A의 output이고 step B의 input이라면, lineage DAG를 자동으로 생성해줍니다.

이와 같은 동작 예시는 [notebook](https://colab.research.google.com/drive/1wZG-jYzPelk8Rs2gIM3a71uEoG46u_nG#scrollTo=DQQVaKS0TmDU)과 해당하는 [W&B Artifacts 페이지](https://wandb.ai/megatruong/metaflow_integration/artifacts/dataset/raw_df/7d14e6578d3f1cfc72fe/graph)에서 확인할 수 있습니다.
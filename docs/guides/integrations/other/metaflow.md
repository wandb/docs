---
title: Metaflow
description: W&B를 Metaflow와 통합하는 방법.
slug: /guides/integrations/metaflow
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

## 개요

[Metaflow](https://docs.metaflow.org)은 ML 워크플로우를 생성하고 실행하기 위해 [Netflix](https://netflixtechblog.com)에서 만든 프레임워크입니다.

이 인테그레이션을 통해 사용자는 Metaflow [steps and flows](https://docs.metaflow.org/metaflow/basics)에 데코레이터를 적용하여 W&B에 파라미터와 아티팩트를 자동으로 로그할 수 있습니다.

* 스텝을 데코레이트하면 해당 스텝 내에서 특정 타입의 로그를 활성화하거나 비활성화할 수 있습니다.
* 플로우를 데코레이트하면 플로우의 모든 스텝에 대해 로그를 활성화하거나 비활성화할 수 있습니다.

## 퀵스타트

### W&B 설치 및 로그인

<Tabs
  defaultValue="notebook"
  values={[
    {label: 'Notebook', value: 'notebook'},
    {label: 'Command Line', value: 'cli'},
  ]}>
  <TabItem value="notebook">

```python
!pip install -Uqqq metaflow fastcore wandb

import wandb
wandb.login()
```
  </TabItem>
  <TabItem value="cli">

```
pip install -Uqqq metaflow fastcore wandb
wandb login
```
  </TabItem>
</Tabs>

### 플로우와 스텝 데코레이션

<Tabs
  defaultValue="step"
  values={[
    {label: 'Step', value: 'step'},
    {label: 'Flow', value: 'flow'},
    {label: 'Flow and Steps', value: 'flow_and_steps'},
  ]}>
  <TabItem value="step">

스텝을 데코레이트하면 해당 스텝 내에서 특정 타입의 로그를 활성화하거나 비활성화할 수 있습니다.

이 예제에서는 모든 datasets와 models가 `start`에서 로그됩니다.

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
  </TabItem>
  <TabItem value="flow">

플로우를 데코레이트하는 것은 기본값으로 모든 구성 요소 스텝을 데코레이트하는 것과 같습니다.

이 경우, `WandbExampleFlow`의 모든 스텝은 datasets와 models를 기본값으로 로그합니다 -- 각 스텝을 `@wandb_log(datasets=True, models=True)`로 데코레이트하는 것과 동일합니다.

```python
from wandb.integration.metaflow import wandb_log

@wandb_log(datasets=True, models=True)  # 모든 @step을 데코레이트 
class WandbExampleFlow(FlowSpec):
    @step
    def start(self):
        self.raw_df = pd.read_csv(...).    # pd.DataFrame -> 데이터셋으로 업로드
        self.model_file = torch.load(...)  # nn.Module    -> 모델로 업로드
        self.next(self.transform)
```
  </TabItem>
  <TabItem value="flow_and_steps">

플로우를 데코레이트하는 것은 기본값으로 모든 스텝을 데코레이트하는 것과 같습니다. 즉, 나중에 특정 스텝을 다른 `@wandb_log`로 데코레이트하면 플로우 레벨의 데코레이션을 덮어씁니다.

아래 예제에서는:

* `start`와 `mid`는 datasets와 models를 로그하지만,
* `end`는 datasets나 models를 로그하지 않습니다.

```python
from wandb.integration.metaflow import wandb_log

@wandb_log(datasets=True, models=True)  # start와 mid를 데코레이트한 것과 동일
class WandbExampleFlow(FlowSpec):
  # 이 스텝은 datasets와 models를 로그합니다.
  @step
  def start(self):
    self.raw_df = pd.read_csv(...).    # pd.DataFrame -> 데이터셋으로 업로드
    self.model_file = torch.load(...)  # nn.Module    -> 모델로 업로드
    self.next(self.mid)

  # 이 스텝도 datasets와 models를 로그합니다.
  @step
  def mid(self):
    self.raw_df = pd.read_csv(...).    # pd.DataFrame -> 데이터셋으로 업로드
    self.model_file = torch.load(...)  # nn.Module    -> 모델로 업로드
    self.next(self.end)

  # 이 스텝은 덮어쓰여져서 datasets나 models를 로그하지 않습니다.
  @wandb_log(datasets=False, models=False)
  @step
  def end(self):
    self.raw_df = pd.read_csv(...).    
    self.model_file = torch.load(...)
```
  </TabItem>
</Tabs>

## 내 데이터는 어디에 있나요? 프로그램적으로 엑세스할 수 있나요?

우리가 캡처한 정보에 엑세스하는 방법은 세 가지가 있습니다: [`wandb` 클라이언트 라이브러리](../../../ref/python/README.md)를 사용하여 원래 Python 프로세스 내부에서 로그된 정보를 엑세스하거나, [웹 앱 UI](../../app/intro.md)를 통해 엑세스하거나, 또는 [우리의 Public API](../../../ref/python/public-api/README.md)를 사용하여 프로그램적으로 엑세스하는 방법입니다. `Parameter`는 W&B의 [`config`](../../track/config.md)에 저장되며 [Overview 탭](../../app/pages/run-page.md#overview-tab)에서 찾을 수 있습니다. `datasets`, `models`, 및 `others`는 [W&B Artifacts](../../artifacts/intro.md)에 저장되며 [Artifacts 탭](../../app/pages/run-page.md#artifacts-tab)에서 찾을 수 있습니다. 기본 Python 타입은 W&B의 [`summary`](../../track/log/intro.md) dict에 저장되며 Overview 탭에서 찾을 수 있습니다. API를 사용하여 외부에서 프로그램적으로 이 정보를 가져오는 방법에 대한 자세한 내용은 우리의 [Public API 가이드](../../track/public-api-guide.md)를 참조하세요.

다음은 요약표입니다:

| 데이터                                         | 클라이언트 라이브러리                      | UI                     |
| --------------------------------------------- | ----------------------------------------- | ---------------------- |
| `Parameter(...)`                              | `wandb.config`                            | Overview 탭, Config    |
| `datasets`, `models`, `others`                | `wandb.use_artifact("{var_name}:latest")` | Artifacts 탭           |
| 기본 Python 타입 (`dict`, `list`, `str`, 등) | `wandb.summary`                           | Overview 탭, Summary   |

### `wandb_log` kwargs

| kwarg      | 옵션                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `datasets` | <ul><li><code>True</code>: 데이터셋인 인스턴스 변수를 로그</li><li><code>False</code></li></ul>                                                                                                                                                                                                                                                                                                                                                                                     |
| `models`   | <ul><li><code>True</code>: 모델인 인스턴스 변수를 로그</li><li><code>False</code></li></ul>                                                                                                                                                                                                                                                                                                                                                                                           |
| `others`   | <ul><li><code>True</code>: 피클로 직렬화할 수 있는 그 외의 모든 것을 로그</li><li><code>False</code></li></ul>                                                                                                                                                                                                                                                                                                                                                                         |
| `settings` | <ul><li><code>wandb.Settings(...)</code>: 이 스텝이나 플로우에 대해 사용자의 <code>wandb</code> 설정 명시</li><li><code>None</code>: <code>wandb.Settings()</code>로 전달하는 것과 동일</li></ul><p>기본적으로, 만약:</p><ul><li><code>settings.run_group</code>이 <code>None</code>이면, <code>\{flow_name\}/\{run_id\}</code>로 설정됩니다.</li><li><code>settings.run_job_type</code>이 <code>None</code>이면, <code>\{run_job_type\}/\{step_name\}</code>로 설정됩니다.</li></ul> |

## 자주 묻는 질문

### 정확히 무엇을 로그합니까? 모든 인스턴스 및 로컬 변수를 로그합니까?

`wandb_log`는 인스턴스 변수만 로그합니다. 로컬 변수는 절대 로그되지 않습니다. 이는 불필요한 데이터를 로그하는 것을 방지하는 데 유용합니다.

### 어떤 데이터 타입을 로그합니까?

현재 우리는 다음 타입을 지원합니다:

| 로그 설정            | 타입                                                                                                                              |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| 기본값 (항상 활성화) | <ul><li><code>dict, list, set, str, int, float, bool</code></li></ul>                                                          |
| `datasets`          | <ul><li><code>pd.DataFrame</code></li><li><code>pathlib.Path</code></li></ul>                                                    |
| `models`            | <ul><li><code>nn.Module</code></li><li><code>sklearn.base.BaseEstimator</code></li></ul>                                         |
| `others`            | <ul><li><a href="https://wiki.python.org/moin/UsingPickle">피클 가능</a>하며 JSON 직렬화 가능인 모든 것</li></ul>                |

### 로그 행동 예제

| 변수 종류        | 행동                          | 예제             | 데이터 타입      |
| ---------------- | ------------------------------ | --------------- | -------------- |
| 인스턴스         | 자동 로그                      | `self.accuracy` | `float`        |
| 인스턴스         | `datasets=True`일 때 로그     | `self.df`       | `pd.DataFrame` |
| 인스턴스         | `datasets=False`일 때 로그 안함 | `self.df`       | `pd.DataFrame` |
| 로컬            | 절대 로그되지 않음             | `accuracy`      | `float`        |
| 로컬            | 절대 로그되지 않음             | `df`            | `pd.DataFrame` |

### 이것이 아티팩트 계보를 추적합니까?

네! 만약 스텝 A의 출력물이고 스텝 B의 입력물인 아티팩트가 있다면, 우리는 자동으로 계보 DAG를 구성합니다.

이 행동의 예제를 보려면, 이 [노트북](https://colab.research.google.com/drive/1wZG-jYzPelk8Rs2gIM3a71uEoG46u_nG#scrollTo=DQQVaKS0TmDU)과 그것에 해당하는 [W&B Artifacts 페이지](https://wandb.ai/megatruong/metaflow_integration/artifacts/dataset/raw_df/7d14e6578d3f1cfc72fe/graph)를 참조하세요.
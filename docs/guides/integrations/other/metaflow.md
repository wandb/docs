---
description: How to integrate W&B with Metaflow.
slug: /guides/integrations/metaflow
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Metaflow

## 개요

[Metaflow](https://docs.metaflow.org)는 [Netflix](https://netflixtechblog.com)에서 ML 워크플로를 생성하고 실행하기 위해 만든 프레임워크입니다.

이 통합은 사용자가 파라미터와 아티팩트를 W&B에 자동으로 로그하기 위해 Metaflow [단계와 플로](https://docs.metaflow.org/metaflow/basics)에 데코레이터를 적용할 수 있게 합니다.

* 단계를 데코레이트하면 해당 단계 내에서 특정 유형에 대한 로깅을 활성화하거나 비활성화할 수 있습니다.
* 플로를 데코레이트하면 플로의 모든 단계에 대해 로깅을 활성화하거나 비활성화할 수 있습니다.

## 퀵스타트

### W&B 설치 및 로그인

<Tabs
  defaultValue="notebook"
  values={[
    {label: '노트북', value: 'notebook'},
    {label: '명령줄', value: 'cli'},
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

### 플로와 단계 데코레이트하기

<Tabs
  defaultValue="step"
  values={[
    {label: '단계', value: 'step'},
    {label: '플로', value: 'flow'},
    {label: '플로 및 단계', value: 'flow_and_steps'},
  ]}>
  <TabItem value="step">

단계를 데코레이트하면 해당 단계 내에서 특정 유형에 대한 로깅을 활성화하거나 비활성화합니다.

이 예에서, `start`의 모든 데이터세트와 모델이 로그됩니다.

```python
from wandb.integration.metaflow import wandb_log

class WandbExampleFlow(FlowSpec):
    @wandb_log(datasets=True, models=True, settings=wandb.Settings(...))
    @step
    def start(self):
        self.raw_df = pd.read_csv(...).    # pd.DataFrame -> 아티팩트로 업로드
        self.model_file = torch.load(...)  # nn.Module    -> 아티팩트로 업로드
        self.next(self.transform)
```
  </TabItem>
  <TabItem value="flow">

플로를 데코레이트하는 것은 모든 구성 단계를 기본값으로 데코레이팅하는 것과 동일합니다.

이 경우, `WandbExampleFlow`의 모든 단계는 기본적으로 데이터세트와 모델을 로그합니다 -- 각 단계를 `@wandb_log(datasets=True, models=True)`로 데코레이팅하는 것과 같습니다.

```python
from wandb.integration.metaflow import wandb_log

@wandb_log(datasets=True, models=True)  # 모든 @step 데코레이트
class WandbExampleFlow(FlowSpec):
    @step
    def start(self):
        self.raw_df = pd.read_csv(...).    # pd.DataFrame -> 아티팩트로 업로드
        self.model_file = torch.load(...)  # nn.Module    -> 아티팩트로 업로드
        self.next(self.transform)
```
  </TabItem>
  <TabItem value="flow_and_steps">

플로를 데코레이트하는 것은 모든 단계를 기본값으로 데코레이팅하는 것과 동일합니다. 즉, 나중에 다른 `@wandb_log`로 단계를 데코레이트하면 플로 수준의 데코레이션을 덮어씁니다.

아래 예에서:

* `start`와 `mid`는 데이터세트와 모델을 로그하지만,
* `end`는 데이터세트나 모델을 로그하지 않습니다.

```python
from wandb.integration.metaflow import wandb_log

@wandb_log(datasets=True, models=True)  # start와 mid를 데코레이트하는 것과 동일
class WandbExampleFlow(FlowSpec):
  # 이 단계는 데이터세트와 모델을 로그합니다.
  @step
  def start(self):
    self.raw_df = pd.read_csv(...).    # pd.DataFrame -> 아티팩트로 업로드
    self.model_file = torch.load(...)  # nn.Module    -> 아티팩트로 업로드
    self.next(self.mid)

  # 이 단계도 데이터세트와 모델을 로그합니다.
  @step
  def mid(self):
    self.raw_df = pd.read_csv(...).    # pd.DataFrame -> 아티팩트로 업로드
    self.model_file = torch.load(...)  # nn.Module    -> 아티팩트로 업로드
    self.next(self.end)

  # 이 단계는 덮어쓰기 되어 데이터세트나 모델을 로그하지 않습니다.
  @wandb_log(datasets=False, models=False)
  @step
  def end(self):
    self.raw_df = pd.read_csv(...).    
    self.model_file = torch.load(...)
```
  </TabItem>
</Tabs>

## 내 데이터는 어디에 있나요? 프로그래매틱하게 엑세스할 수 있나요?

[`wandb` 클라이언트 라이브러리](../../../ref/python/README.md)를 사용하여 로그되는 원래 Python 프로세스 내부, [웹 앱 UI](../../app/intro.md)를 통해, 또는 [공개 API](../../../ref/python/public-api/README.md)를 사용하여 프로그래매틱하게 정보에 엑세스할 수 있습니다. `Parameter`들은 W&B의 [`config`](../../track/config.md)에 저장되며, [Overview 탭](../../app/pages/run-page.md#overview-tab)에서 찾을 수 있습니다. `datasets`, `models`, `others`는 [W&B Artifacts](../../artifacts/intro.md)에 저장되며, [Artifacts 탭](../../app/pages/run-page.md#artifacts-tab)에서 찾을 수 있습니다. 기본 파이썬 유형은 W&B의 [`summary`](../../track/log/intro.md) 딕셔너리에 저장되며, Overview 탭에서 찾을 수 있습니다. API를 사용하여 외부에서 프로그래매틱하게 이 정보를 얻는 방법에 대한 자세한 내용은 [공개 API 가이드](../../track/public-api-guide.md)를 참조하세요.

여기에 요약이 있습니다:

| 데이터                                           | 클라이언트 라이브러리                           | UI                   |
| ----------------------------------------------- | --------------------------------------- | -------------------- |
| `Parameter(...)`                                | `wandb.config`                          | Overview 탭, 설정    |
| `datasets`, `models`, `others`                  | `wandb.use_artifact("{var_name}:latest")` | Artifacts 탭        |
| 기본 Python 유형 (`dict`, `list`, `str`, 등.) | `wandb.summary`                         | Overview 탭, 요약    |

### `wandb_log` kwargs

| 키워드      | 옵션                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `datasets` | <ul><li><code>True</code>: 데이터세트인 인스턴스 변수를 로그합니다</li><li><code>False</code></li></ul>                                                                                                                                                                                                                                                                                                                                                                        |
| `models`   | <ul><li><code>True</code>: 모델인 인스턴스 변수를 로그합니다</li><li><code>False</code></li></ul>                                                                                                                                                                                                                                                                                                                                                                              |
| `others`   | <ul><li><code>True</code>: 피클로 직렬화 가능하고 JSON 직렬화 가능한 그 외 모든 것을 로그합니다</li><li><code>False</code></li></ul>                                                                                                                                                                                                                                                                                                                                         |
| `settings` | <ul><li><code>wandb.Settings(...)</code>: 이 단계나 플로에 대한 자신만의 <code>wandb</code> 설정을 지정합니다</li><li><code>None</code>: <code>wandb.Settings()</code>을 전달하는 것과 동일합니다</li></ul><p>기본적으로:</p><ul><li><code>settings.run_group</code>이 <code>None</code>이면, <code>{flow_name}/{run_id}</code>로 설정됩니다</li><li><code>settings.run_job_type</code>이 <code>None</code>이면, <code>{run_job_type}/{step_name}</code>로 설정됩니다</li></ul> |

## 자주 묻는 질문

### 정확히 무엇을 로그하나요? 모든 인스턴스와 로컬 변수를 로그하나요?

`wandb_log`는 인스턴스 변수만 로그합니다. 로컬 변수는 절대 로그되지 않습니다. 이는 불필요한 데이터를 로깅하지 않도록 하기 위해 유용합니다.

### 어떤 데이터 유형이 로그되나요?

현재 이러한 유형을 지원합니다:

| 로깅 설정            | 유형                                                                                                                    |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| 기본값 (항상 켜짐) | <ul><li><code>dict, list, set, str, int, float, bool</code></li></ul>                                                   |
| `datasets`          | <ul><li><code>pd.DataFrame</code></li><li><code>pathlib.Path</code></li></ul>                                           |
| `models`            | <ul><li><code>nn.Module</code></li><li><code>sklearn.base.BaseEstimator</code></li></ul>                                |
| `others`            | <ul><li>[피클로 직렬화 가능](https://wiki.python.org/moin/UsingPickle)하고 JSON 직렬화 가능한 모든 것</li></ul>      |

### 로깅 동작 예시

| 변수 종류          | 동작                         | 예시              | 데이터 유형      |
| ---------------- | --------------------------- | ----------------- | -------------- |
| 인스턴스         | 자동 로그됨                  | `self.accuracy`   | `float`        |
| 인스턴스         | `datasets=True`일 때 로그됨 | `self.df`         | `pd.DataFrame` |
| 인스턴스         | `datasets=False`일 때 로그 안 됨 | `self.df`      | `pd.DataFrame` |
| 로컬            | 절대 로그되지 않음            | `accuracy`        | `float`        |
| 로컬            | 절대 로그되지 않음            | `df`              | `pd.DataFrame` |

### 아티팩트 계보를 추적하나요?

네! 단계 A의 출력물이자 단계 B의 입력물인 아티팩트가 있으면, 자동으로 계보 DAG를 구성해 줍니다.

이 동작의 예시는 이 [노트북](https://colab.research.google.com/drive/1wZG-jYzPelk8Rs2gIM3a71uEoG46u_nG#scrollTo=DQQVaKS0TmDU)과 해당 [W&B Artifacts 페이지](https://wandb.ai/megatruong/metaflow_integration/artifacts/dataset/raw_df/7d14e6578d3f1cfc72fe/graph)를 참조하세요.
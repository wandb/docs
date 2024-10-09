---
title: Sweeps FAQ
description: W&B Sweeps에 대한 자주 묻는 질문에 대한 답변.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

### W&B Sweep의 일부로 모든 하이퍼파라미터의 값을 제공해야 하나요? 기본값을 설정할 수 있나요?

스윕 구성의 일부로 명시된 하이퍼파라미터 이름과 값은 `wandb.config`에 액세스할 수 있으며, 이는 사전과 같은 오브젝트입니다.

스윕의 일부가 아닌 run의 경우, `wandb.config`의 값은 보통 `wandb.init`의 `config` 인수에 사전을 제공하여 설정합니다. 그러나 스윕 중에는 `wandb.init`에 전달된 설정 정보가 기본값으로 처리되며, 이는 스윕에 의해 덮어씌워질 수 있습니다.

또한, `config.setdefaults`를 사용하여 의도된 행동을 명확히 할 수 있습니다. 두 메서드에 대한 코드조각은 아래에 나타납니다:

<Tabs
  defaultValue="wandb.init"
  values={[
    {label: 'wandb.init', value: 'wandb.init'},
    {label: 'config.setdefaults', value: 'config.setdef'},
  ]}>
  <TabItem value="wandb.init">

```python
# 하이퍼파라미터의 기본값 설정
config_defaults = {"lr": 0.1, "batch_size": 256}

# run 시작, 스윕에 의해 덮어씌워질 수 있는 기본값 제공
with wandb.init(config=config_defaults) as run:
    # 여기에 트레이닝 코드를 추가하세요
    ...
```

  </TabItem>
  <TabItem value="config.setdef">

```python
# 하이퍼파라미터의 기본값 설정
config_defaults = {"lr": 0.1, "batch_size": 256}

# run 시작
with wandb.init() as run:
    # 스윕에 의해 설정되지 않은 값을 업데이트
    run.config.setdefaults(config_defaults)

    # 여기에 트레이닝 코드를 추가하세요
```

  </TabItem>
</Tabs>

### SLURM에서 스윕을 어떻게 실행해야 하나요?

[SLURM 스케줄링 시스템](https://slurm.schedmd.com/documentation.html)을 사용할 때, 각각의 예약된 작업에서 `wandb agent --count 1 SWEEP_ID`를 실행하는 것을 권장합니다. 이는 단일 트레이닝 작업을 실행한 후 종료됩니다. 이는 자원을 요청할 때 런타임을 예측하기 쉽게 하며, 하이퍼파라미터 검색의 병렬성을 활용합니다.

### 그리드 검색을 다시 실행할 수 있나요?

예. 그리드 검색을 완료했지만 W&B Runs를 일부 다시 실행하고 싶다면(예를 들어, 일부가 충돌했을 경우), 다시 실행하고 싶은 W&B Runs를 삭제하고, [스윕 제어 페이지](./sweeps-ui.md)에서 **재개** 버튼을 선택합니다. 마지막으로, 새로운 스윕 ID로 새로운 W&B Sweep 에이전트를 시작합니다.

완료된 W&B Runs의 파라미터 조합은 다시 실행되지 않습니다.

### 스윕에서 사용자 정의 CLI 명령어를 사용할 수 있나요?

커맨드라인 인수를 전달하여 트레이닝의 일부 요소를 설정하는 경우, 사용자 정의 CLI 명령어와 함께 W&B Sweeps를 사용할 수 있습니다.

예를 들어, 다음 코드조각은 사용자가 train.py라는 파이썬 스크립트를 트레이닝하는 bash 터미널을 보여줍니다. 사용자는 값을 전달하여 파이썬 스크립트에서 이를 구문 분석합니다:

```bash
/usr/bin/env python train.py -b \
    your-training-config \
    --batchsize 8 \
    --lr 0.00001
```

사용자 정의 명령어를 사용하려면 YAML 파일의 `command` 키를 편집하세요. 위 예제를 계속해서 설명하자면, 다음과 같이 보일 수 있습니다:

```yaml
program:
  train.py
method: grid
parameters:
  batch_size:
    value: 8
  lr:
    value: 0.0001
command:
  - ${env}
  - python
  - ${program}
  - "-b"
  - your-training-config
  - ${args}
```

`${args}` 키는 인수로 구문 분석할 수 있도록 스윕 구성 파일의 모든 파라미터로 확장됩니다: `argparse: --param1 value1 --param2 value2`

`argparse`로 지정하고 싶지 않은 추가 인수가 있는 경우 다음을 사용할 수 있습니다:

```python
parser = argparse.ArgumentParser()
args, unknown = parser.parse_known_args()
```

:::info
환경에 따라 `python`이 파이썬 2를 가리킬 수 있습니다. 파이썬 3을 호출하려면 `python3`를 사용하여 명령어를 설정하세요:

```yaml
program:
  script.py
command:
  - ${env}
  - python3
  - ${program}
  - ${args}
```
:::

### 스윕에 추가 값을 추가할 수 있는 방법이 있나요, 아니면 새로 시작해야 하나요?

W&B Sweep이 시작되면 Sweep 구성을 변경할 수 없습니다. 그러나 테이블 보기에서 run을 선택할 수 있도록 체크박스를 사용하고, **스윕 생성** 메뉴 옵션을 사용하여 이전 run을 사용하여 새로운 Sweep 구성을 생성할 수 있습니다.

### 하이퍼파라미터로 불리언 변수를 플래그할 수 있나요?

하이퍼파라미터를 불리언 플래그로 전달하려면 구성의 커맨드 섹션에서 `${args_no_boolean_flags}` 매크로를 사용할 수 있습니다. 이는 자동으로 모든 불리언 파라미터를 플래그로 전달합니다. `param`이 `True`인 경우 명령어는 `--param`을 받으며, `param`이 `False`인 경우 플래그가 생략됩니다.

### Sweeps와 SageMaker를 사용할 수 있나요?

예. 간단히 말해, W&B를 인증해야 하며, 내장된 SageMaker 추정기를 사용하는 경우 `requirements.txt` 파일을 생성해야 합니다. 인증 및 `requirements.txt` 파일 설정 방법에 대한 자세한 내용은 [SageMaker 인테그레이션](../integrations/other/sagemaker.md) 가이드를 참조하세요.

:::info
완전한 예제는 [GitHub](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cifar10-sagemaker)에서 확인할 수 있으며, [블로그](https://wandb.ai/site/articles/running-sweeps-with-sagemaker)에서도 자세히 읽어볼 수 있습니다.\
SageMaker와 W&B를 사용하여 감정 분석기를 배포하는 [튜토리얼](https://wandb.ai/authors/sagemaker/reports/Deploy-Sentiment-Analyzer-Using-SageMaker-and-W-B--VmlldzoxODA1ODE)도 읽어보실 수 있습니다.
:::

### AWS Batch, ECS 등의 클라우드 인프라에서 W&B Sweeps를 사용할 수 있나요?

일반적으로, 잠재적인 W&B Sweep 에이전트가 읽을 수 있는 위치에 `sweep_id`를 게시하는 방법과 이러한 Sweep 에이전트가 이 `sweep_id`를 소비하고 run을 시작하는 방법이 필요합니다.

다시 말해서, `wandb agent`를 호출할 수 있는 무언가가 필요합니다. 예를 들어, EC2 인스턴스를 가져온 후 `wandb agent`를 호출합니다. 이 경우, SQS 큐를 사용하여 `sweep_id`를 여러 EC2 인스턴스에 방송하고, 큐에서 `sweep_id`를 소비하고 run을 시작합니다.

### 내 스윕 로그를 로컬로 기록할 디렉토리를 변경할 수 있나요?

`WANDB_DIR`라는 환경 변수 설정을 통해 W&B가 run 데이터를 기록할 디렉토리의 경로를 변경할 수 있습니다. 예를 들어:

```python
os.environ["WANDB_DIR"] = os.path.abspath("your/directory")
```

### 여러 메트릭 최적화하기

동일한 run에서 여러 메트릭을 최적화하려면 개별 메트릭의 가중치 합을 사용할 수 있습니다.

```python
metric_combined = 0.3 * metric_a + 0.2 * metric_b + ... + 1.5 * metric_n
wandb.log({"metric_combined": metric_combined})
```

새로운 결합 메트릭을 로그하고 이를 최적화 목표로 설정합니다:

```yaml
metric:
  name: metric_combined
  goal: minimize
```

### 스윕에서 코드 로깅을 활성화하려면 어떻게 해야 하나요?

스윕에서 코드 로깅을 활성화하려면, W&B Run을 초기화한 후 `wandb.log_code()`를 추가하세요. 이는 W&B 프로필의 앱 설정 페이지에서 코드 로깅을 활성화했더라도 필요합니다. 코드 로깅에 대한 자세한 내용은 [`wandb.log_code()` 문서 여기를](../../ref/python/run.md#log_code) 참조하세요.

### "Est. Runs" 열이란 무엇인가요?

W&B는 이산 검색 공간을 사용하여 W&B Sweep을 생성할 때 발생할 것으로 예상되는 run 수를 제공합니다. 총 run 수는 검색 공간의 카테시안 곱입니다.

예를 들어, 다음 검색 공간을 제공한다고 가정해보겠습니다:

![](/images/sweeps/sweeps_faq_whatisestruns_1.png)

이 예에서 카테시안 곱은 9입니다. W&B는 이 숫자를 W&B 앱 UI에 예상 run 수(**Est. Runs**)로 표시합니다:

![](/images/sweeps/spaces_sweeps_faq_whatisestruns_2.webp)

W&B SDK를 사용하여 예상되는 Run 수를 얻을 수도 있습니다. Sweep 오브젝트의 `expected_run_count` 속성을 사용하여 예상 Run 수를 얻으세요:

```python
sweep_id = wandb.sweep(
    sweep_configs, project="your_project_name", entity="your_entity_name"
)
api = wandb.Api()
sweep = api.sweep(f"your_entity_name/your_project_name/sweeps/{sweep_id}")
print(f"EXPECTED RUN COUNT = {sweep.expected_run_count}")
```
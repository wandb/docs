---
description: Answers to frequently asked question about W&B Sweeps.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# FAQ

<head>
  <title>스윕에 대한 자주 묻는 질문들</title>
</head>

### 모든 하이퍼파라미터에 대해 값을 제공해야 하나요? 기본값을 설정할 수 있나요?

스윕 구성의 일부로 지정된 하이퍼파라미터 이름과 값은 `wandb.config`, 사전과 유사한 객체에서 접근 가능합니다.

스윕의 일부가 아닌 실행의 경우 `wandb.config`의 값은 `wandb.init`의 `config` 인수에 사전을 제공하여 보통 설정됩니다. 그러나 스윕 중에는 `wandb.init`에 전달된 모든 구성 정보가 대신 기본값으로 처리되며, 이는 스윕에 의해 덮어쓰일 수 있습니다.

보다 명시적으로 의도된 동작에 대해 `config.setdefaults`을 사용할 수도 있습니다. 아래에 두 가지 방법에 대한 코드 조각이 나와 있습니다:

<Tabs
  defaultValue="wandb.init"
  values={[
    {label: 'wandb.init', value: 'wandb.init'},
    {label: 'config.setdefaults', value: 'config.setdef'},
  ]}>
  <TabItem value="wandb.init">

```python
# 하이퍼파라미터에 대한 기본값 설정
config_defaults = {"lr": 0.1, "batch_size": 256}

# 기본값을 제공하여 실행 시작
#   스윕에 의해 덮어쓰일 수 있음
with wandb.init(config=config_default) as run:
    # 여기에 학습 코드를 추가하세요
    ...
```

  </TabItem>
  <TabItem value="config.setdef">

```python
# 하이퍼파라미터에 대한 기본값 설정
config_defaults = {"lr": 0.1, "batch_size": 256}

# 실행 시작
with wandb.init() as run:
    # 스윕에 의해 설정되지 않은 값 업데이트
    run.config.setdefaults(config_defaults)

    # 여기에 학습 코드를 추가하세요
```

  </TabItem>
</Tabs>

### SLURM에서 스윕을 실행하는 방법은 무엇인가요?

[SLURM 스케줄링 시스템](https://slurm.schedmd.com/documentation.html)을 사용하여 스윕을 실행할 때, 각 예약된 작업에서 `wandb agent --count 1 SWEEP_ID`를 실행하는 것이 좋습니다. 이렇게 하면 단일 학습 작업을 실행한 후 종료되므로, 리소스를 요청할 때 런타임을 더 쉽게 예측할 수 있고 하이퍼파라미터 검색의 병렬성을 활용할 수 있습니다.

### 그리드 검색을 다시 실행할 수 있나요?

네. 그리드 검색을 완료했지만 W&B 실행 중 일부를 다시 실행하고 싶은 경우(예: 일부가 충돌한 경우)가 있습니다. 다시 실행하려는 W&B 실행을 삭제한 다음, [스윕 제어 페이지](./sweeps-ui.md)에서 **다시 시작** 버튼을 선택하세요. 마지막으로 새 스윕 ID로 새 W&B 스윕 에이전트를 시작합니다.

완료된 W&B 실행을 가진 파라미터 조합은 다시 실행되지 않습니다.

### 스윕과 함께 사용자 정의 CLI 명령을 어떻게 사용하나요?

학습을 구성하는 데 명령줄 인수를 전달하여 일부 측면을 정상적으로 구성하는 경우 W&B 스윕과 사용자 정의 CLI 명령을 사용할 수 있습니다.

예를 들어, 다음 코드 조각은 사용자가 train.py라는 Python 스크립트를 학습하는 bash 터미널을 보여줍니다. 사용자는 파이썬 스크립트 내에서 구문 분석될 값들을 전달합니다:

```bash
/usr/bin/env python train.py -b \
    your-training-config \
    --batchsize 8 \
    --lr 0.00001
```

사용자 정의 명령을 사용하려면 YAML 파일의 `command` 키를 편집하세요. 예를 들어, 위의 예제를 계속하면 다음과 같습니다:

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

`${args}` 키는 스윕 구성 파일의 모든 파라미터를 `argparse: --param1 value1 --param2 value2`에 의해 구문 분석될 수 있도록 확장합니다.

`argparse`와 함께 지정하고 싶지 않은 추가 인수가 있는 경우 사용할 수 있습니다:

```python
parser = argparse.ArgumentParser()
args, unknown = parser.parse_known_args()
```

:::info
환경에 따라 `python`은 Python 2를 가리킬 수 있습니다. 명령을 구성할 때 Python 3이 호출되도록 하려면 `python` 대신 `python3`를 사용하세요:

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

### 스윕에 추가 값을 추가할 수 있나요, 아니면 새로운 스윕을 시작해야 하나요?

W&B 스윕이 시작되면 스윕 구성을 변경할 수 없습니다. 그러나 모든 테이블 뷰에서 체크박스를 사용하여 실행을 선택한 다음, **스윕 생성** 메뉴 옵션을 사용하여 이전 실행을 사용하여 새 스윕 구성을 생성할 수 있습니다.

### 부울 변수를 하이퍼파라미터로 플래그 지정할 수 있나요?

`config` 섹션의 명령에서 `${args_no_boolean_flags}` 매크로를 사용하여 하이퍼파라미터를 부울 플래그로 전달할 수 있습니다. 이렇게 하면 모든 부울 파라미터가 플래그로 자동 전달됩니다. `param`이 `True`일 때 명령은 `--param`을 받고, `param`이 `False`일 때 플래그는 생략됩니다.

### Sweeps와 SageMaker를 사용할 수 있나요?

네. 한눈에 보기에, W&B를 인증하고 내장된 SageMaker 추정기를 사용하는 경우 `requirements.txt` 파일을 생성해야 합니다. 인증 방법과 `requirements.txt` 파일 설정 방법에 대한 자세한 내용은 [SageMaker 통합](../integrations/other/sagemaker.md) 가이드를 참조하세요.

:::info
완전한 예제는 [GitHub](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cifar10-sagemaker)에서 사용할 수 있으며, [블로그](https://wandb.ai/site/articles/running-sweeps-with-sagemaker)에서 SageMaker와 함께 스윕을 실행하는 방법에 대해 자세히 알아볼 수 있습니다.\
SageMaker와 W&B를 사용하여 감정 분석기를 배포하는 방법에 대한 [튜토리얼](https://wandb.ai/authors/sagemaker/reports/Deploy-Sentiment-Analyzer-Using-SageMaker-and-W-B--VmlldzoxODA1ODE)도 읽어보세요.
:::

### AWS Batch, ECS 등과 같은 클라우드 인프라와 W&B 스윕을 사용할 수 있나요?

일반적으로, 잠재적인 W&B 스윕 에이전트가 읽을 수 있는 위치에 `sweep_id`를 게시하고, 이러한 스윕 에이전트가 이 `sweep_id`를 소비하고 실행을 시작할 수 있는 방법이 필요합니다.

다시 말해, `wandb agent`를 호출할 수 있는 무언가가 필요합니다. 예를 들어, EC2 인스턴스를 올린 다음 그것에서 `wandb agent`를 호출하는 경우가 있을 수 있습니다. 이 경우 SQS 큐를 사용하여 몇 개의 EC2 인스턴스에 `sweep_id`를 방송한 다음 이들이 큐에서 `sweep_id`를 소비하고 실행을 시작하도록 할 수 있습니다.

### 스윕 로그를 로컬로 디렉터리를 변경하는 방법은 무엇인가요?

환경 변수 `WANDB_DIR`을 설정하여 W&B가 실행 데이터를 로그할 디렉터리 경로를 변경할 수 있습니다. 예를 들어:

```python
os.environ["WANDB_DIR"] = os.path.abspath("your/directory")
```

### 다중 메트릭 최적화

동일한 실행에서 여러 메트릭을 최적화하고 싶다면, 개별 메트릭의 가중합을 사용할 수 있습니다.

```python
metric_combined = 0.3 * metric_a + 0.2 * metric_b + ... + 1.5 * metric_n
wandb.log({"metric_combined": metric_combined})
```

새로운 결합된 메트릭을 로그하고 최적화 목표로 설정하세요:

```yaml
metric:
  name: metric_combined
  goal: minimize
```

### 스윕과 함께 코드 로깅을 활성화하는 방법은 무엇인가요?

스윕에 대한 코드 로깅을 활성화하려면, W&B 실행을 초기화한 후에 `wandb.log_code()`를 추가하기만 하면 됩니다. 이는 앱의 W&B 프로필 설정 페이지에서 코드 로깅을 활성화했을 때에도 필요합니다. 보다 고급 코드 로깅에 대해서는 [여기에서 `wandb.log_code()` 문서](../../ref/python/run.md#log_code)를 참조하세요.

### "Est. Runs" 열은 무엇인가요?

W&B는 이산 검색 공간으로 W&B 스윕을 생성할 때 발생할 것으로 예상되는 실행 수를 제공합니다. 실행의 총 수는 검색 공간의 카테시안 곱입니다.

예를 들어, 다음과 같은 검색 공간을 제공한다고 가정해 보세요:

![](/images/sweeps/sweeps_faq_whatisestruns_1.png)

이 예에서는 카테시안 곱이 9입니다. W&B는 W&B 앱 UI에서 예상 실행 횟수(**Est. Runs**)로 이 숫자를 보여줍니다:

![](/images/sweeps/spaces_sweeps_faq_whatisestruns_2.webp)

W&B SDK를 사용하여 예상 실행 횟수도 얻을 수 있습니다. Sweep 객체의 `expected_run_count` 속성을 사용하여 예상 실행 횟수를 얻으세요:

```python
sweep_id = wandb.sweep(
    sweep_configs, project="your_project_name", entity="your_entity_name"
)
api = wandb.Api()
sweep = api.sweep(f"your_entity_name/your_project_name/sweeps/{sweep_id}")
print(f"EXPECTED RUN COUNT = {sweep.expected_run_count}")
```
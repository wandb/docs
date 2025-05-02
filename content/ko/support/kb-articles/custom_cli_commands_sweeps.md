---
title: How do I use custom CLI commands with sweeps?
menu:
  support:
    identifier: ko-support-kb-articles-custom_cli_commands_sweeps
support:
- sweeps
toc_hide: true
type: docs
url: /ko/support/:filename
translationKey: custom_cli_commands_sweeps
---
만약 트레이닝 설정이 코맨드 라인 인수를 전달한다면, 사용자 정의 CLI 코맨드와 함께 W&B Sweeps를 사용할 수 있습니다.

아래 예제에서, 코드조각은 사용자가 `train.py`라는 Python 스크립트를 트레이닝하고 스크립트가 파싱하는 값을 제공하는 bash 터미널을 보여줍니다:

```bash
/usr/bin/env python train.py -b \
    your-training-config \
    --batchsize 8 \
    --lr 0.00001
```

사용자 정의 코맨드를 구현하려면 YAML 파일에서 `command` 키를 수정하십시오. 이전 예제를 기반으로, 설정은 다음과 같이 나타납니다:

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

`${args}` 키는 스윕 구성의 모든 파라미터를 `--param1 value1 --param2 value2` 와 같이 `argparse`용으로 포맷하여 확장합니다.

`argparse` 외부의 추가 인수의 경우, 다음을 구현하십시오:

```python
parser = argparse.ArgumentParser()
args, unknown = parser.parse_known_args()
```

{{% alert %}}
환경에 따라, `python`은 Python 2를 참조할 수 있습니다. Python 3의 호출을 보장하려면, 코맨드 설정에서 `python3`를 사용하십시오:

```yaml
program:
  script.py
command:
  - ${env}
  - python3
  - ${program}
  - ${args}
```
{{% /alert %}}

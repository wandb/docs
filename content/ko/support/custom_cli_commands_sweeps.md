---
title: How do I use custom CLI commands with sweeps?
menu:
  support:
    identifier: ko-support-custom_cli_commands_sweeps
tags:
- sweeps
toc_hide: true
type: docs
---

만약 트레이닝 설정이 코맨드라인 인수를 전달한다면, 사용자 정의 CLI 코맨드와 함께 W&B Sweeps 를 사용할 수 있습니다.

아래 예시에서, 코드 조각은 `train.py` 라는 Python 스크립트를 트레이닝하는 bash 터미널을 보여주고, 스크립트가 파싱하는 값들을 제공합니다:

```bash
/usr/bin/env python train.py -b \
    your-training-config \
    --batchsize 8 \
    --lr 0.00001
```

사용자 정의 코맨드를 구현하려면, YAML 파일에서 `command` 키를 수정하세요. 이전 예제를 기반으로 하면, 설정은 다음과 같이 나타납니다:

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

`${args}` 키는 스윕 구성의 모든 파라미터를 확장하여 `argparse` 에 대해 `--param1 value1 --param2 value2` 와 같이 포맷합니다.

`argparse` 외부의 추가적인 인수의 경우, 다음을 구현합니다:

```python
parser = argparse.ArgumentParser()
args, unknown = parser.parse_known_args()
```

{{% alert %}}
환경에 따라, `python` 은 Python 2를 참조할 수 있습니다. Python 3의 호출을 보장하려면, 코맨드 설정에서 `python3` 를 사용하세요:

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

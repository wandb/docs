---
title: 스윕에서 커스텀 CLI 코맨드를 어떻게 사용하나요?
menu:
  support:
    identifier: ko-support-kb-articles-custom_cli_commands_sweeps
support:
- 스윕
toc_hide: true
type: docs
url: /support/:filename
---

W&B Sweeps 는 트레이닝 설정이 명령줄 인수로 전달되는 경우, 커스텀 CLI 코맨드와 함께 사용할 수 있습니다.

아래 예시 코드조각에서는 사용자가 `train.py`라는 Python 스크립트를 트레이닝하는 bash 터미널을 보여주며, 스크립트가 파싱할 값을 지정하는 모습입니다:

```bash
/usr/bin/env python train.py -b \
    your-training-config \
    --batchsize 8 \
    --lr 0.00001
```

커스텀 코맨드를 적용하려면 YAML 파일에서 `command` 키를 수정하면 됩니다. 앞선 예시를 적용한 구성은 아래와 같습니다:

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

`${args}` 키는 스윕 구성에 있는 모든 파라미터를 `argparse` 형식인 `--param1 value1 --param2 value2`로 확장합니다.

추가적인 인수가 `argparse` 외부에 있을 경우, 아래와 같이 구현합니다:

```python
parser = argparse.ArgumentParser()
args, unknown = parser.parse_known_args() # 알려지지 않은 인수도 파싱
```

{{% alert %}}
환경에 따라 `python`이 Python 2를 가리킬 수도 있습니다. Python 3를 사용해야 한다면, 코맨드 설정에서 `python3`를 사용하세요:

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
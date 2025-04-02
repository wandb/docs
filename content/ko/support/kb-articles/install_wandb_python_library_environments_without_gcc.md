---
title: How do I install the wandb Python library in environments without gcc?
menu:
  support:
    identifier: ko-support-kb-articles-install_wandb_python_library_environments_without_gcc
support:
- python
toc_hide: true
type: docs
url: /support/:filename
---

`wandb` 설치 시 다음과 같은 오류가 발생하는 경우:

```
unable to execute 'gcc': No such file or directory
error: command 'gcc' failed with exit status 1
```

사전 빌드된 휠에서 직접 `psutil`을 설치합니다. [https://pywharf.github.io/pywharf-pkg-repo/psutil](https://pywharf.github.io/pywharf-pkg-repo/psutil) 에서 Python 버전과 운영 체제를 확인하세요.

예를 들어, Linux에서 Python 3.8에 `psutil`을 설치하려면 다음과 같이 하세요.

```bash
WHEEL_URL=https://github.com/pywharf/pywharf-pkg-repo/releases/download/psutil-5.7.0-cp38-cp38-manylinux2010_x86_64.whl#sha256=adc36dabdff0b9a4c84821ef5ce45848f30b8a01a1d5806316e068b5fd669c6d
pip install $WHEEL_URL
```

`psutil`을 설치한 후, `pip install wandb`를 실행하여 `wandb` 설치를 완료합니다.

---
title: gcc가 없는 환경에서 wandb Python 라이브러리를 어떻게 설치하나요?
menu:
  support:
    identifier: ko-support-kb-articles-install_wandb_python_library_environments_without_gcc
support:
- 파이썬
toc_hide: true
type: docs
url: /support/:filename
---

`wandb` 를 설치할 때 다음과 같은 오류가 발생한다면:

```
unable to execute 'gcc': No such file or directory
error: command 'gcc' failed with exit status 1
```

사전 빌드된 wheel 파일에서 `psutil` 을 직접 설치하세요. 사용 중인 Python 버전과 운영체제를 [https://pywharf.github.io/pywharf-pkg-repo/psutil](https://pywharf.github.io/pywharf-pkg-repo/psutil) 에서 확인하세요.

예를 들어, Linux 환경의 Python 3.8 에서 `psutil` 을 설치하려면 다음과 같이 실행합니다:

```bash
# wheel 파일의 다운로드 URL을 설정합니다
WHEEL_URL=https://github.com/pywharf/pywharf-pkg-repo/releases/download/psutil-5.7.0-cp38-cp38-manylinux2010_x86_64.whl#sha256=adc36dabdff0b9a4c84821ef5ce45848f30b8a01a1d5806316e068b5fd669c6d
pip install $WHEEL_URL
```

`psutil` 이 설치된 후, `pip install wandb` 를 실행하여 `wandb` 설치를 마무리하세요.
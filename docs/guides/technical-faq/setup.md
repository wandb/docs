---
title: Setup FAQ
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

### 내 트레이닝 코드에서 run의 이름을 어떻게 설정할 수 있나요?

트레이닝 스크립트 상단에서 `wandb.init`을 호출할 때 다음과 같이 experiment의 이름을 전달하세요: `wandb.init(name="my_awesome_run")`.

### wandb를 오프라인으로 실행할 수 있나요?

오프라인 머신에서 트레이닝을 수행하고 나중에 결과를 서버에 업로드하고자 한다면, 이를 위한 기능이 있습니다!

1. 환경 변수 `WANDB_MODE=offline`을 설정하면 인터넷 연결 없이 메트릭을 로컬에 저장할 수 있습니다.
2. 준비가 되면 디렉토리에서 `wandb init`을 실행하여 프로젝트 이름을 설정하세요.
3. `wandb sync YOUR_RUN_DIRECTORY`를 실행하여 메트릭을 클라우드 서비스로 푸시하고 호스팅된 웹 앱에서 결과를 확인하세요.

API를 사용하여 `run.settings._offline` 또는 `run.settings.mode`를 통해 run이 오프라인인지 확인할 수 있습니다.

#### [`wandb sync`](../../ref/cli/wandb-sync.md)를 사용할 수 있는 몇 가지 사례

* 인터넷이 없는 경우.
* 모든 기능을 완전히 비활성화해야 하는 경우.
* 여러 가지 이유로 나중에 run을 동기화하려는 경우. 예를 들어, 트레이닝 머신에서 리소스를 사용하는 것을 피하고 싶다면.

### 이것은 파이썬에서만 작동하나요?

현재, 라이브러리는 Python 2.7+ 및 3.6+ 프로젝트와 함께 작동합니다. 위에 언급된 아키텍처를 통해 다른 언어와 쉽게 통합할 수 있을 것입니다. 다른 언어의 모니터링이 필요하다면 [contact@wandb.com](mailto:contact@wandb.com)으로 메일을 보내주세요.

### 아나콘다 패키지가 있나요?

네! `pip` 또는 `conda`로 설치할 수 있습니다. 후자의 경우, [conda-forge](https://conda-forge.org) 채널에서 패키지를 가져와야 합니다.

<Tabs
  defaultValue="pip"
  values={[
    {label: 'pip', value: 'pip'},
    {label: 'conda', value: 'conda'},
  ]}>
  <TabItem value="pip">

```bash
# conda 환경 생성
conda create -n wandb-env python=3.8 anaconda
# 생성된 환경 활성화
conda activate wandb-env
# 이 conda 환경에서 pip으로 wandb 설치
pip install wandb
```

  </TabItem>
  <TabItem value="conda">

```
conda activate myenv
conda install wandb --channel conda-forge
```

  </TabItem>
</Tabs>

이 설치에 문제가 있으면 알려주세요. 이 Anaconda [패키지 관리 도큐멘트](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html)에 유용한 가이드가 있습니다.

### gcc가 없는 환경에서 wandb Python 라이브러리를 어떻게 설치하나요?

`wandb`를 설치하려고 할 때 다음과 같은 오류가 나타날 수 있습니다:

```
unable to execute 'gcc': No such file or directory
error: command 'gcc' failed with exit status 1
```

이 경우 `psutil`을 사전 빌드된 wheel에서 직접 설치할 수 있습니다. Python 버전과 운영 체제를 여기를 통해 찾으세요: [https://pywharf.github.io/pywharf-pkg-repo/psutil](https://pywharf.github.io/pywharf-pkg-repo/psutil)

예를 들어, Linux에서 Python 3.8에 `psutil`을 설치하려면:

```bash
WHEEL_URL=https://github.com/pywharf/pywharf-pkg-repo/releases/download/psutil-5.7.0-cp38-cp38-manylinux2010_x86_64.whl/psutil-5.7.0-cp38-cp38-manylinux2010_x86_64.whl#sha256=adc36dabdff0b9a4c84821ef5ce45848f30b8a01a1d5806316e068b5fd669c6d
pip install $WHEEL_URL
```

`psutil`이 설치된 후 `pip install wandb`를 통해 wandb를 설치할 수 있습니다.

### W&B 클라이언트는 Python 2를 지원하나요? <a href="#eol-python27" id="eol-python27"></a>

W&B 클라이언트 라이브러리는 버전 0.10까지 Python 2.7 및 Python 3을 지원했습니다. Python 2의 수명 종료로 인해 Python 2.7의 지원은 버전 0.11부터 중단되었습니다. Python 2.7 시스템에서 `pip install --upgrade wandb`를 실행하는 사용자는 0.10.x 시리즈의 새로운 릴리스를 받을 수 있습니다. 0.10.x 시리즈에 대한 지원은 중요한 버그 수정 및 패치로 제한됩니다. 현재 0.10.33 버전이 0.10.x 시리즈의 마지막으로 Python 2.7을 지원하는 버전입니다.

### W&B 클라이언트는 Python 3.5를 지원하나요? <a href="#eol-python35" id="eol-python35"></a>

W&B 클라이언트 라이브러리는 버전 0.11까지 Python 3.5를 지원했습니다. Python 3.5의 수명 종료로 인해, [버전 0.12](https://github.com/wandb/wandb/releases/tag/v0.12.0)부터 지원이 중단되었습니다.
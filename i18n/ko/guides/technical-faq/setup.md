---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 설치하기

### 학습 스크립트에서 실행 이름을 어떻게 설정할 수 있나요?

학습 스크립트 상단에서 `wandb.init`을 호출할 때, 다음과 같이 실험 이름을 전달하세요: `wandb.init(name="my_awesome_run")`.

### wandb를 오프라인으로 실행할 수 있나요?

오프라인 기계에서 학습을 하고 나중에 결과를 서버에 업로드하고 싶다면, 저희가 그런 기능을 제공합니다!

1. 환경 변수 `WANDB_MODE=offline`을 설정하여 로컬에 메트릭을 저장하세요, 인터넷이 필요하지 않습니다.
2. 준비가 되었을 때, 프로젝트 이름을 설정하기 위해 디렉터리에서 `wandb init`을 실행하세요.
3. `wandb sync YOUR_RUN_DIRECTORY`를 실행하여 메트릭을 클라우드 서비스로 전송하고 호스팅된 웹 앱에서 결과를 확인하세요.

`wandb.init()` 후에 `run.settings._offline` 또는 `run.settings.mode`를 사용하여 API를 통해 실행이 오프라인인지 확인할 수 있습니다.

#### [`wandb sync`](../../ref/cli/wandb-sync.md)을 사용할 수 있는 몇 가지 사례

* 인터넷이 없을 때.
* 모든 것을 완전히 비활성화해야 할 때.
* 어떤 이유로 나중에 실행을 동기화해야 할 때. 예를 들어, 학습 기계에서 리소스를 사용하는 것을 피하고 싶을 때.

### 이것은 Python에만 작동하나요?

현재, 라이브러리는 Python 2.7+ 및 3.6+ 프로젝트에서만 작동합니다. 위에 언급된 아키텍처는 다른 언어와 쉽게 통합할 수 있게 해야 합니다. 다른 언어의 모니터링이 필요하다면 [contact@wandb.com](mailto:contact@wandb.com)으로 메시지를 보내주세요.

### 아나콘다 패키지가 있나요?

네! `pip` 또는 `conda`로 설치할 수 있습니다. 후자의 경우, [conda-forge](https://conda-forge.org) 채널에서 패키지를 받아야 합니다.

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
# 이 conda 환경에서 pip로 wandb 설치
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


이 설치와 관련하여 문제가 발생하면 알려주세요. 이 아나콘다 [패키지 관리 문서](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html)에 도움이 될 만한 가이드가 있습니다.

### gcc가 없는 환경에서 wandb Python 라이브러리를 어떻게 설치하나요?

`wandb`를 설치하려고 하다가 다음과 같은 오류를 보게 된다면:

```
unable to execute 'gcc': No such file or directory
error: command 'gcc' failed with exit status 1
```

사전 빌드된 휠에서 직접 `psutil`을 설치할 수 있습니다. 여기서 Python 버전과 OS를 찾아보세요: [https://pywharf.github.io/pywharf-pkg-repo/psutil](https://pywharf.github.io/pywharf-pkg-repo/psutil)

예를 들어, Linux에서 Python 3.8에 `psutil`을 설치하는 경우:

```bash
WHEEL_URL=https://github.com/pywharf/pywharf-pkg-repo/releases/download/psutil-5.7.0-cp38-cp38-manylinux2010_x86_64.whl/psutil-5.7.0-cp38-cp38-manylinux2010_x86_64.whl#sha256=adc36dabdff0b9a4c84821ef5ce45848f30b8a01a1d5806316e068b5fd669c6d
pip install $WHEEL_URL
```

`psutil`이 설치된 후, `pip install wandb`로 wandb를 설치할 수 있습니다.

### W&B 클라이언트는 Python 2를 지원하나요? <a href="#eol-python27" id="eol-python27"></a>

W&B 클라이언트 라이브러리는 버전 0.10까지 Python 2.7 및 Python 3를 모두 지원했습니다. Python 2의 수명이 끝남에 따라, 버전 0.11부터 Python 2.7에 대한 지원이 중단되었습니다. Python 2.7 시스템에서 `pip install --upgrade wandb`를 실행하는 사용자는 0.10.x 시리즈의 새 릴리스만 받게 됩니다. 0.10.x 시리즈에 대한 지원은 중요한 버그 수정과 패치로 제한될 것입니다. 현재, Python 2.7을 지원하는 0.10.x 시리즈의 마지막 버전은 0.10.33입니다.

### W&B 클라이언트는 Python 3.5를 지원하나요? <a href="#eol-python35" id="eol-python35"></a>

W&B 클라이언트 라이브러리는 버전 0.11까지 Python 3.5를 지원했습니다. Python 3.5의 수명이 끝남에 따라, [버전 0.12](https://github.com/wandb/wandb/releases/tag/v0.12.0)부터 지원이 중단되었습니다.
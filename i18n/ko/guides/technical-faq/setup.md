---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 설정

### 트레이닝 코드에서 run의 이름을 어떻게 구성할 수 있나요?

트레이닝 스크립트의 상단에서 `wandb.init`를 호출할 때, 실험명을 전달하세요. 예: `wandb.init(name="my_awesome_run")`.

### wandb를 오프라인으로 실행할 수 있나요?

오프라인 머신에서 트레이닝을 하고 나중에 결과를 우리 서버에 업로드하고 싶다면, 우리에게는 당신을 위한 기능이 있습니다!

1. 환경 변수 `WANDB_MODE=offline`을 설정하여 인터넷이 필요 없이 로컬에 메트릭을 저장합니다.
2. 준비가 되었을 때, 프로젝트 이름을 설정하기 위해 디렉토리에서 `wandb init`을 실행합니다.
3. `wandb sync YOUR_RUN_DIRECTORY`를 실행하여 메트릭을 우리 클라우드 서비스에 푸시하고 호스티드 웹 앱에서 결과를 확인합니다.

API를 사용하여 `wandb.init()` 후에 run이 오프라인인지 확인할 수 있습니다. `run.settings._offline` 또는 `run.settings.mode`를 사용하세요.

#### [`wandb sync`](../../ref/cli/wandb-sync.md)를 사용할 수 있는 몇 가지 사례

* 인터넷이 없는 경우.
* 모든 것을 완전히 비활성화해야 하는 경우.
* 어떤 이유로 나중에 run을 동기화해야 하는 경우. 예: 트레이닝 머신의 리소스 사용을 피하고 싶은 경우.

### 이것은 파이썬에만 작동하나요?

현재, 라이브러리는 파이썬 2.7+ & 3.6+ 프로젝트에서만 작동합니다. 위에서 언급한 아키텍처는 다른 언어와 쉽게 통합할 수 있게 해야 합니다. 만약 다른 언어의 모니터링이 필요하다면, [contact@wandb.com](mailto:contact@wandb.com)으로 메모를 보내주세요.

### 아나콘다 패키지가 있나요?

네! `pip`으로 설치하거나 `conda`로 설치할 수 있습니다. 후자의 경우, [conda-forge](https://conda-forge.org) 채널에서 패키지를 가져와야 합니다.

<Tabs
  defaultValue="pip"
  values={[
    {label: 'pip', value: 'pip'},
    {label: 'conda', value: 'conda'},
  ]}>
  <TabItem value="pip">

```bash
# 콘다 환경 생성
conda create -n wandb-env python=3.8 anaconda
# 생성된 환경 활성화
conda activate wandb-env
# 이 콘다 환경에서 wandb를 pip로 설치
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


이 설치에서 문제가 발생하면, 알려주세요. 이 아나콘다 [패키지 관리 문서](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html)에 도움이 될만한 가이드가 있습니다.

### gcc 없는 환경에서 wandb 파이썬 라이브러리를 어떻게 설치하나요?

`wandb`를 설치하려고 시도하다가 이 오류를 보게 되면:

```
unable to execute 'gcc': No such file or directory
error: command 'gcc' failed with exit status 1
```

사전 빌드된 휠에서 직접 `psutil`을 설치할 수 있습니다. 여기에서 파이썬 버전과 OS를 찾으세요: [https://pywharf.github.io/pywharf-pkg-repo/psutil](https://pywharf.github.io/pywharf-pkg-repo/psutil)

예를 들어, 리눅스의 파이썬 3.8에서 `psutil`을 설치하려면:

```bash
WHEEL_URL=https://github.com/pywharf/pywharf-pkg-repo/releases/download/psutil-5.7.0-cp38-cp38-manylinux2010_x86_64.whl/psutil-5.7.0-cp38-cp38-manylinux2010_x86_64.whl#sha256=adc36dabdff0b9a4c84821ef5ce45848f30b8a01a1d5806316e068b5fd669c6d
pip install $WHEEL_URL
```

`psutil`이 설치되면, `pip install wandb`로 wandb를 설치할 수 있습니다.

### W&B 클라이언트는 Python 2를 지원하나요? <a href="#eol-python27" id="eol-python27"></a>

W&B 클라이언트 라이브러리는 버전 0.10까지 Python 2.7과 Python 3을 모두 지원했습니다. Python 2의 생명 종료로 인해, 버전 0.11부터 Python 2.7에 대한 지원이 중단되었습니다. Python 2.7 시스템에서 `pip install --upgrade wandb`를 실행하는 사용자는 0.10.x 시리즈의 새로운 릴리스만 받게 됩니다. 0.10.x 시리즈에 대한 지원은 중요한 버그 수정 및 패치로 제한됩니다. 현재, 버전 0.10.33은 Python 2.7을 지원하는 0.10.x 시리즈의 마지막 버전입니다.

### W&B 클라이언트는 Python 3.5를 지원하나요? <a href="#eol-python35" id="eol-python35"></a>

W&B 클라이언트 라이브러리는 버전 0.11까지 Python 3.5를 지원했습니다. Python 3.5의 생명 종료로 인해, [버전 0.12](https://github.com/wandb/wandb/releases/tag/v0.12.0)부터 지원이 중단되었습니다.
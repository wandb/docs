---
title: Manage algorithms locally
description: W&B 클라우드 호스팅 서비스를 사용하는 대신 로컬에서 검색 및 중지 알고리즘을 실행하십시오.
displayed_sidebar: default
---

기본적으로 하이퍼파라미터 컨트롤러는 Weights & Biases에서 클라우드 서비스로 호스팅됩니다. W&B 에이전트는 컨트롤러와 통신하여 트레이닝에 사용할 다음 파라미터 세트를 결정합니다. 컨트롤러는 또한 조기 중지 알고리즘을 실행하여 어떤 run을 중지할 수 있는지 결정하는 역할을 합니다.

로컬 컨트롤러 기능을 사용하면 사용자가 검색을 시작하고 알고리즘을 로컬에서 중단할 수 있습니다. 로컬 컨트롤러는 사용자가 문제를 디버깅하고 클라우드 서비스에 통합할 수 있는 새로운 기능을 개발하기 위해 코드 점검 및 도구 활용을 가능하게 합니다.

:::caution
이 기능은 Sweeps 툴의 새로운 알고리즘을 더 빠르게 개발하고 디버깅을 지원하기 위해 제공됩니다. 실제 하이퍼파라미터 최적화 워크로드를 위한 것이 아닙니다.
:::

시작하기 전에 W&B SDK(`wandb`)를 설치해야 합니다. 다음 코드조각을 커맨드라인에 입력하세요:

pip install wandb sweeps 

다음 예시는 이미 설정 파일과 트레이닝 루프가 Python 스크립트 또는 Jupyter Notebook에 정의되어 있다고 가정합니다. 설정 파일을 정의하는 방법에 대한 자세한 정보는 [Define sweep configuration](./define-sweep-configuration.md)을 참조하십시오.

### 커맨드라인에서 로컬 컨트롤러 실행하기

W&B의 클라우드 서비스로 호스팅되는 하이퍼파라미터 컨트롤러를 사용할 때처럼 스윕을 초기화하세요. 로컬 컨트롤러를 W&B 스윕 작업에 사용하고자 한다는 것을 표시하기 위해 컨트롤러 플래그 (`controller`)를 지정하십시오:

wandb sweep --controller config.yaml

또는 스윕 초기화와 로컬 컨트롤러를 사용하고자 한다는 것의 지정을 두 단계로 분리할 수 있습니다.

단계를 분리하려면 먼저 스윕의 YAML 설정 파일에 다음의 키-값을 추가하세요:

controller:
  type: local

다음은 스윕을 초기화하세요:

wandb sweep config.yaml

스윕이 초기화된 후, [`wandb controller`](../../ref/python/controller.md)로 컨트롤러를 시작하세요:

# wandb sweep 명령어는 스윕 ID를 출력합니다
wandb controller {user}/{entity}/{sweep_id}

로컬 컨트롤러 사용을 지정했다면, 스윕을 실행하기 위해 하나 이상의 스윕 에이전트를 시작하세요. W&B 스윕을 보통처럼 시작하십시오. 자세한 정보는 [Start sweep agents](../../guides/sweeps/start-sweep-agents.md)를 참조하십시오.

wandb sweep sweep_ID

### W&B Python SDK와 함께 로컬 컨트롤러 실행하기

다음 코드조각은 W&B Python SDK를 사용하여 로컬 컨트롤러를 지정하고 사용하는 방법을 보여줍니다.

Python SDK에서 컨트롤러를 사용하는 가장 간단한 방법은 스윕 ID를 [`wandb.controller`](../../ref/python/controller.md) 메소드에 전달하는 것입니다. 그 다음, 반환된 오브젝트의 `run` 메소드를 사용하여 스윕 작업을 시작하세요:

sweep = wandb.controller(sweep_id)
sweep.run()

컨트롤러 루프에 대한 더 많은 제어가 필요하다면:

import wandb

sweep = wandb.controller(sweep_id)
while not sweep.done():
    sweep.print_status()
    sweep.step()
    time.sleep(5)

제공된 파라미터에 대해 더 많은 제어가 필요하다면:

import wandb

sweep = wandb.controller(sweep_id)
while not sweep.done():
    params = sweep.search()
    sweep.schedule(params)
    sweep.print_status()

코드만으로 전체 스윕을 지정하고자 한다면, 다음과 같이 할 수 있습니다:

import wandb

sweep = wandb.controller()
sweep.configure_search("grid")
sweep.configure_program("train-dummy.py")
sweep.configure_controller(type="local")
sweep.configure_parameter("param1", value=3)
sweep.create()
sweep.run()
---
description: Run Weights and Biases on your own machines using Docker
displayed_sidebar: default
---

# 기본 설정

Docker를 사용하여 자신의 기계에서 Weights and Biases를 실행하세요.

### 설치

[Docker](https://www.docker.com)와 [Python](https://www.python.org)이 설치된 모든 기계에서 다음을 실행하십시오:

```
pip install wandb
wandb server start
```

### 로그인

처음 로그인하는 경우 로컬 W&B 서버 계정을 생성하고 API 키를 인증해야 합니다. 여러 대의 기계에서 `wandb`를 실행하거나 프라이빗 인스턴스와 W&B 클라우드 간에 전환할 때 실행 위치를 제어하는 몇 가지 방법이 있습니다.

공유 프라이빗 인스턴스에 메트릭을 보내려면 아래 절차를 따르십시오. 이미 DNS를 설정했는지 확인하십시오:

1. 로그인할 때마다 호스트 플래그를 프라이빗 인스턴스의 주소로 설정하십시오:

```bash
wandb login --host=http://wandb.your-shared-local-host.com
```

2. 환경 변수 `WANDB_BASE_URL`을 로컬 인스턴스의 주소로 설정하십시오:

```bash
export WANDB_BASE_URL="http://wandb.your-shared-local-host.com"
```

자동화된 환경에서는 `WANDB_API_KEY`를 설정할 수 있습니다. 키는 [wandb.your-shared-local-host.com/authorize](http://wandb.your-shared-local-host.com/authorize)에서 찾을 수 있습니다.

호스트를 `api.wandb.ai`로 설정하여 W&B의 공용 클라우드 인스턴스에 로그를 기록하십시오:

```bash
wandb login --cloud
```

또는

```bash
export WANDB_BASE_URL="https://api.wandb.ai"
```

브라우저에서 클라우드 호스팅된 wandb 계정에 로그인한 상태에서 [https://wandb.ai/settings](https://wandb.ai/settings)에서 클라우드 API 키로 전환할 수도 있습니다.

### 무료 라이선스 생성

W&B 서버를 구성을 완료하려면 라이선스가 필요합니다. [**Deploy Manager 열기**](https://deploy.wandb.ai/deploy)에서 무료 라이선스를 생성하십시오. 클라우드 계정이 없는 경우 무료 라이선스를 생성하기 위해 계정을 생성해야 합니다. 개인 또는 팀 또는 무료 라이선스를 생성할 수 있습니다:

1. [**개인 라이선스**](https://deploy.wandb.ai/deploy)는 개인 작업을 위해 영원히 무료입니다: ![](/images/hosting/personal_license.png)
2. [**팀 시행 라이선스**](https://deploy.wandb.ai/deploy)는 30일 동안 무료이며, 팀을 설정하고 확장 가능한 백엔드를 연결할 수 있습니다: ![](/images/hosting/team_trial_license.png)

### 로컬 호스트에 라이선스 추가

1. Deployment에서 라이선스를 복사하고 W&B 서버의 localhost로 돌아가십시오: ![](/images/hosting/add_license_local_host.png)
2. localhost의 `/system-admin` 페이지에 붙여넣어 로컬 설정에 추가하십시오:
   ![](@site/static/images/hosting/License.gif)

### 업그레이드

_wandb/local_의 새 버전은 정기적으로 DockerHub에 푸시됩니다. 버전을 최신 상태로 유지하는 것이 좋습니다. 업그레이드하려면 다음 명령을 터미널에 복사하여 붙여넣으십시오:

```shell
$ wandb server start --upgrade
```

또는, 인스턴스를 수동으로 업그레이드할 수 있습니다. 다음 코드 조각을 터미널에 복사하여 붙여넣으십시오:

```shell
$ docker pull wandb/local
$ docker stop wandb-local
$ docker run --rm -d -v wandb:/vol -p 8080:8080 --name wandb-local wandb/local
```

### 지속성 및 확장성

- W&B 서버로 보낸 모든 메타데이터와 파일은 `/vol` 디렉터리에 저장됩니다. 이 위치에 영구 볼륨을 마운트하지 않으면 docker 프로세스가 종료될 때 모든 데이터가 손실됩니다.
- 이 솔루션은 [프로덕션](../hosting-options/intro.md) 작업을 위한 것이 아닙니다.
- 메타데이터를 외부 MySQL 데이터베이스에 저장하고 파일을 외부 스토리지 버킷에 저장할 수 있습니다.
- 기본 파일 저장소는 크기를 조정할 수 있어야 합니다. 최소 저장 공간 임계값이 초과되면 기본 파일 시스템의 크기를 조정하기 위해 알림을 설정해야 합니다.
- 엔터프라이즈 시행의 경우, 이미지/비디오/오디오가 많지 않은 작업 부하에 대해 기본 볼륨에 최소 100GB의 여유 공간이 있어야 합니다.

#### wandb는 사용자 계정 데이터를 어떻게 지속합니까?

Kubernetes 인스턴스가 중지되면, W&B 애플리케이션은 모든 사용자 계정 데이터를 tarball로 번들링하여 Amazon S3 오브젝트 스토어에 업로드합니다. 인스턴스를 다시 시작하고 `BUCKET` 환경 변수를 제공할 때 W&B는 이전에 업로드된 tarball 파일을 가져옵니다. W&B는 또한 새로 시작된 Kubernetes 배포에 사용자 계정 정보를 로드합니다.

외부 오브젝트 스토어가 활성화되면 모든 사용자 데이터를 포함하므로 강력한 엑세스 제어를 시행해야 합니다.
W&B는 외부 버킷에 인스턴스 설정을 지속합니다. W&B는 또한 인증서 및 비밀을 버킷에 지속합니다.

#### 공유 인스턴스 생성 및 확장

W&B의 강력한 협업 기능을 즐기려면 중앙 서버에 공유 인스턴스가 필요합니다. 이는 [AWS, GCP, Azure, Kubernetes 또는 Docker에서 설정할 수 있습니다](../hosting-options/intro.md).

:::caution
**시행 모드 대비 프로덕션 설정**

W&B Local의 시행 모드에서는 단일 기계에서 Docker 컨테이너를 실행합니다. 이 설정은 제품을 테스트하기에는 이상적이지만 확장 가능하지 않습니다.

프로덕션 작업을 위해 데이터 손실을 피하기 위해 확장 가능한 파일 시스템을 설정하십시오. 우리는 당신이:
* 사전에 추가 공간을 할당하고,
* 더 많은 데이터를 로그로 기록함에 따라 파일 시스템을 선제적으로 크기 조정하고
* 백업을 위해 외부 메타데이터 및 오브젝트 스토어를 구성하십시오.

공간이 부족하면 인스턴스가 작동을 멈추고 그 시점 이후의 어떠한 추가 데이터도 손실됩니다.
:::

W&B 서버의 Enterprise 옵션에 대해 자세히 알아보려면 [영업팀에 문의하십시오](https://wandb.ai/site/contact).
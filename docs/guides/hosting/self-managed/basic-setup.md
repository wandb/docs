---
title: Tutorial: Run a W&B Server using Docker
description: 자체 머신에서 Docker를 사용하여 Weights and Biases 실행하기
displayed_sidebar: default
---

"Hello, world!" 예제를 따라 W&B 서버를 전용 클라우드 및 자체 관리 호스팅 옵션에 설치하는 일반 워크플로우를 배우세요. 이 데모가 끝나면 트라이얼 모드 W&B 라이센스를 사용하여 로컬 머신에서 W&B 서버를 호스팅하는 방법을 알게 될 것입니다.

이 데모에서는 `8080` 포트(`localhost:8080`)에서 로컬 개발 서버를 사용합니다.

:::tip
**트라이얼 모드 대 프로덕션 설정**

트라이얼 모드에서는 Docker 컨테이너를 단일 머신에서 실행합니다. 이 설정은 제품을 테스트하는 데 이상적이지만 확장 가능하지는 않습니다.

프로덕션 작업을 위해서는 데이터 손실을 피하기 위해 확장 가능한 파일 시스템을 설정해야 합니다. W&B는 다음을 강력히 권장합니다:
* 사전에 여유 공간을 할당하고,
* 더 많은 데이터를 기록할 때 파일 시스템을 적극적으로 크기 조정하고,
* 백업을 위해 외부 메타데이터 및 오브젝트 저장소를 구성하십시오.
:::

## 사전 요구 사항
시작하기 전에 로컬 머신이 다음 요구 사항을 충족하는지 확인하세요: 

1. [Python](https://www.python.org) 설치
2. [Docker](https://www.docker.com) 설치 및 실행 확인
3. 최신 버전의 W&B 설치 또는 업그레이드:
   ```bash
   pip install --upgrade wandb
   ```
## 1. W&B Docker 이미지 가져오기

다음 명령을 터미널에서 실행하세요:

```bash
wandb server start
```

이 코맨드는 최신 W&B Docker 이미지 [`wandb/local`](https://hub.docker.com/r/wandb/local) 를 가져옵니다.

## 2. W&B 계정 만들기
`http://localhost:8080/signup`으로 이동하여 초기 사용자 계정을 생성합니다. 이름, 이메일 주소, 사용자 이름 및 비밀번호를 제공합니다:

![](/images/hosting/signup_localhost.png)

**회원 가입** 버튼을 클릭하여 W&B 계정을 생성합니다.

:::note
이 데모를 위해 W&B 계정이 이미 있더라도 새 W&B 계정을 생성하세요.
:::

### API 키 복사하기
계정을 만든 후 `http://localhost:8080/authorize`로 이동합니다.

화면에 나타나는 W&B API 키를 복사하세요. 나중 단계에서 이 키가 로그인 자격 증명을 확인하는 데 필요합니다.

![](/images/hosting/copy_api_key.png)

## 3. 라이센스 생성하기
https://deploy.wandb.ai/deploy 에 있는 W&B 배포 관리자에 가서 트라이얼 모드 W&B 라이센스를 생성하세요.

1. Docker를 제공자로 선택하세요.
![](/images/hosting/deploy_manager_platform.png)
2. **다음**을 클릭하세요.
3. **라이센스의 소유자** 드롭다운에서 라이센스 소유자를 선택하세요.
![](/images/hosting/deploy_manager_info.png)
4. **다음** 버튼을 클릭하세요.
5. **인스턴스 이름** 필드에 라이센스 이름을 제공하세요.
6. (선택 사항) **설명** 필드에 라이센스에 대한 설명을 제공하세요.
7. **라이센스 키 생성** 버튼을 클릭하세요.
![](/images/hosting/deploy_manager_generate.png)

**라이센스 키 생성** 버튼을 클릭하면 W&B가 배포 라이센스 페이지로 리디렉션합니다. 배포 라이센스 페이지에서 배포 ID, 라이센스가 속한 조직 등의 정보를 볼 수 있습니다.

:::tip
특정 라이센스 인스턴스를 보는 두 가지 방법 중 하나를 선택하세요:
1. 배포 관리자 UI로 이동하여 라이센스 인스턴스 이름을 클릭하세요.
2. `https://deploy.wandb.ai/DeploymentID`로 직접 이동하여, `DeploymentID`는 라이센스 인스턴스에 할당된 고유 ID입니다.
:::

## 4. 로컬 호스트에 트라이얼 라이센스 추가하기
1. 라이센스 인스턴스의 배포 라이센스 페이지에서 **라이센스 복사** 버튼을 클릭합니다.
![](/images/hosting/deploy_manager_get_license.png)
2. `http://localhost:8080/system-admin/`으로 이동합니다.
3. 라이센스를 **라이센스 필드**에 붙여 넣습니다.
![](/images/hosting/License.gif)
4. **설정 업데이트** 버튼을 클릭합니다.

## 5. 브라우저가 W&B 앱 UI를 실행 중인지 확인하기
W&B가 로컬 컴퓨터에서 실행 중인지 확인하세요. `http://localhost:8080/home`으로 이동하세요. 브라우저에서 W&B 앱 UI가 보여야 합니다.

![](/images/hosting/check_local_host.png)

## 6. 로컬 W&B 인스턴스에 프로그램 접근 추가하기

1. `http://localhost:8080/authorize`로 이동하여 API 키를 얻습니다.
2. 터미널에서 다음을 실행하세요:
   ```bash
   wandb login --host=http://localhost:8080/
   ```
   이미 다른 계정으로 W&B에 로그인되어 있는 경우, `relogin` 플래그를 추가하세요:
   ```bash
   wandb login --relogin --host=http://localhost:8080
   ```
3. 요청 시 API 키를 입력하세요.

W&B는 자동 로그인을 위해 여러분의 .netrc 프로필 `/Users/username/.netrc`에 `localhost` 프로필과 API 키를 추가합니다.

## 데이터를 보존하기 위한 볼륨 추가하기

로그된 모든 메타데이터와 파일은 `https://deploy.wandb.ai/vol` 디렉토리에 임시 저장됩니다.

파일과 메타데이터를 로컬 W&B 인스턴스에 저장하려면 Docker 컨테이너에 볼륨 또는 외부 저장소를 마운트하세요. W&B는 메타데이터를 외부 MySQL 데이터베이스에 저장하고 파일을 Amazon S3와 같은 외부 저장 버킷에 저장할 것을 권장합니다.

:::info
트라이얼 W&B 라이센스를 사용하여 생성된 로컬 W&B 인스턴스는 Docker를 사용하여 로컬 브라우저에서 W&B를 실행합니다. 기본적으로 Docker 컨테이너가 더 이상 존재하지 않으면 데이터가 보존되지 않습니다. `https://deploy.wandb.ai/vol`에 볼륨을 마운트하지 않으면 Docker 프로세스가 죽을 때 데이터가 손실됩니다.
:::

볼륨을 마운트하고 Docker가 데이터를 관리하는 방법에 대한 자세한 내용은 Docker 문서의 [Manage data in Docker](https://docs.docker.com/storage/) 페이지를 참조하세요.

### 볼륨 고려사항
기본 파일 저장소는 크기를 조절할 수 있어야 합니다. 
W&B는 최소 저장 용량 임계값에 가까워졌을 때 이를 알려주는 알림을 설정하여 기본 파일 시스템의 크기를 조정할 수 있도록 할 것을 권장합니다.

:::info
기업용 트라이얼의 경우, W&B는 이미지/비디오/오디오가 아닌 무거운 워크로드에 대해 볼륨에 최소 100GB의 여유 공간을 권장합니다.
:::

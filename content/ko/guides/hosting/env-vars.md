---
title: 환경 변수 구성하기
description: W&B 서버 설치 구성 방법
menu:
  default:
    identifier: ko-guides-hosting-env-vars
    parent: w-b-platform
weight: 7
---

System Settings 관리 UI를 통해 인스턴스 레벨의 설정을 구성하는 것 외에도, W&B에서는 환경 변수(Environment Variables)를 사용하여 코드로도 해당 값을 설정할 수 있습니다. IAM에 대한 고급 설정은 [IAM 고급 설정]({{< relref path="./iam/advanced_env_vars.md" lang="ko" >}})도 참고하세요.

## 환경 변수 레퍼런스

| 환경 변수                         | 설명                                                                                                                                                                       |
|------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `LICENSE`                          | wandb/local 라이선스                                                                                                                                                       |
| `MYSQL`                            | MySQL 연결 문자열                                                                                                                                                          |
| `BUCKET`                           | 데이터 저장용 S3 / GCS 버킷                                                                                                                                                |
| `BUCKET_QUEUE`                     | 오브젝트 생성 이벤트를 위한 SQS / Google PubSub 큐                                                                                                                          |
| `NOTIFICATIONS_QUEUE`              | run 이벤트를 발행하는 SQS 큐                                                                                                                                                |
| `AWS_REGION`                       | 버킷이 위치한 AWS 리전                                                                                                                                                     |
| `HOST`                             | 인스턴스의 FQDN, 예: `https://my.domain.net`                                                                                                                                |
| `OIDC_ISSUER`                      | Open ID Connect 아이덴티티 제공자의 URL, 예: `https://cognito-idp.us-east-1.amazonaws.com/us-east-1_uiIFNdacd`                                                          |
| `OIDC_CLIENT_ID`                   | 아이덴티티 제공자에서 애플리케이션의 Client ID                                                                                                                              |
| `OIDC_AUTH_METHOD`                 | Implicit(기본값) 또는 pkce, 자세한 내용은 아래 참고                                                                                                                         |
| `SLACK_CLIENT_ID`                  | 알림에 사용할 Slack 애플리케이션의 client ID                                                                                                                               |
| `SLACK_SECRET`                     | 알림에 사용할 Slack 애플리케이션의 secret                                                                                                                                    |
| `LOCAL_RESTORE`                    | 인스턴스에 엑세스할 수 없는 경우 일시적으로 true로 설정할 수 있습니다. 임시 인증 정보는 컨테이너의 로그에서 확인하세요.                                                       |
| `REDIS`                            | 외부 REDIS 인스턴스를 W&B와 연동할 때 사용할 수 있습니다.                                                                                                                   |
| `LOGGING_ENABLED`                  | true로 설정하면 엑세스 로그가 stdout으로 스트리밍됩니다. 이 변수를 설정하지 않고도 사이드카 컨테이너를 마운트하여 `/var/log/gorilla.log`을 tail 할 수 있습니다.               |
| `GORILLA_ALLOW_USER_TEAM_CREATION` | true로 설정하면 비관리자 사용자도 새로운 팀을 생성할 수 있습니다. 기본값은 false입니다.                                                                                        |
| `GORILLA_CUSTOMER_SECRET_STORE_SOURCE` | 팀의 secret을 저장하기 위한 시크릿 매니저를 설정합니다. W&B Weave에서 사용됩니다. 지원되는 시크릿 매니저는 다음과 같습니다: <ul><li><b>내부 시크릿 매니저</b> (기본값): <code>k8s-secretmanager://wandb-secret</code></li><li><b>AWS Secret Manager</b>: <code>aws-secretmanager</code></li><li><b>GCP Secret Manager</b>: <code>gcp-secretmanager</code></li><li><b>Azure</b>: <code>az-secretmanger</code></li></ul> |
| `GORILLA_DATA_RETENTION_PERIOD`    | run에서 삭제된 데이터를 몇 시간 동안 보관할지 설정합니다. 삭제된 run 데이터는 복구할 수 없습니다. 입력값 끝에 `h`를 붙입니다. 예: `"24h"`                                |
| `GORILLA_DISABLE_PERSONAL_ENTITY`  | true로 설정하면 [personal entities]({{< relref path="/support/kb-articles/difference_team_entity_user_entity_mean_me.md" lang="ko" >}}) 기능이 꺼집니다. 새 personal 프로젝트 생성과 기존 personal 프로젝트에의 쓰기를 방지합니다. |
| `ENABLE_REGISTRY_UI`               | true로 설정하면 새로운 W&B Registry UI를 활성화합니다.                                                                                           |
| `WANDB_ARTIFACT_DIR`               | 다운로드한 모든 아티팩트가 저장되는 위치입니다. 설정하지 않으면 트레이닝 스크립트 기준 상대경로의 `artifacts` 디렉토리가 기본값입니다. 해당 디렉토리가 존재하고 실행 중인 사용자가 쓸 수 있는 권한이 있어야 합니다. 이 변수는 생성된 메타데이터 파일의 위치는 제어하지 않습니다. 메타데이터 파일의 위치는 `WANDB_DIR` 환경변수로 따로 지정할 수 있습니다.             |
| `WANDB_DATA_DIR`                   | 임시 아티팩트를 업로드할 위치입니다. 기본 경로는 사용 중인 플랫폼에 따라 다르며, `platformdirs` Python 패키지의 `user_data_dir` 값을 사용합니다. 해당 디렉토리가 존재하고 실행 중인 사용자가 쓸 수 있는 권한이 있어야 합니다. |
| `WANDB_DIR`                        | 생성된 모든 파일을 저장할 위치입니다. 설정하지 않으면 트레이닝 스크립트 기준 상대경로의 `wandb` 디렉토리가 기본값입니다. 해당 디렉토리가 존재하고 실행 중인 사용자가 쓸 수 있는 권한이 있어야 합니다. 이 변수는 다운로드된 아티팩트의 위치는 제어하지 않습니다. 아티팩트의 위치는 `WANDB_ARTIFACT_DIR` 환경변수로 따로 지정할 수 있습니다.             |
| `WANDB_IDENTITY_TOKEN_FILE`        | [아이덴티티 연합]({{< relref path="/guides/hosting/iam/authentication/identity_federation.md" lang="ko" >}})을 위해 Java Web Token (JWT)이 저장된 로컬 디렉토리의 절대경로입니다.                                                               |


{{% alert %}}
`GORILLA_DATA_RETENTION_PERIOD` 환경 변수는 신중하게 사용하세요. 해당 변수를 설정한 순간 데이터가 즉시 삭제됩니다. 이 플래그를 활성화하기 전 반드시 데이터베이스와 스토리지 버킷을 백업하는 것을 권장합니다.
{{% /alert %}}

## 고급 신뢰성 설정

### Redis

외부 Redis 서버 설정은 필수는 아니지만, 프로덕션 환경에서는 강력하게 권장합니다. Redis는 서비스의 신뢰성을 높이고, 특히 대규모 프로젝트에서 로딩 시간을 줄이기 위해 캐싱을 가능하게 합니다. 아래와 같은 사양을 갖춘 관리형 Redis 서비스(예: 고가용성(HA)이 적용된 ElastiCache)를 추천합니다.

- 최소 4GB 메모리, 권장 8GB
- Redis 버전 6.x
- 전송 중 암호화
- 인증 활성화

W&B에서 Redis 인스턴스를 설정하려면 `http(s)://YOUR-W&B-SERVER-HOST/system-admin`의 W&B 설정 페이지로 이동합니다. "외부 Redis 인스턴스 사용" 옵션을 활성화한 뒤, 아래와 같은 형식으로 Redis 연결 문자열을 입력하세요.

{{< img src="/images/hosting/configure_redis.png" alt="W&B에서 REDIS 설정하기" >}}

환경 변수 `REDIS`를 컨테이너나 Kubernetes 배포 환경에서 지정해 Redis를 설정할 수도 있습니다. 또는 `REDIS`를 Kubernetes 시크릿으로 구성할 수도 있습니다.

이 문서에서는 Redis 인스턴스가 기본 포트인 `6379`에서 실행된다고 가정합니다. 다른 포트로 설정했거나 인증, TLS 활성화를 원한다면 연결 문자열 포맷은 다음과 같습니다: `redis://$USER:$PASSWORD@$HOST:$PORT?tls=true`

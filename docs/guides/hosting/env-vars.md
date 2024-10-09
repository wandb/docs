---
title: Configure environment variables
description: W&B 서버 설치를 구성하는 방법
displayed_sidebar: default
---

시스템 설정 관리자 UI를 통해 인스턴스 레벨 설정을 구성하는 것 외에도, W&B는 환경 변수(Environment Variables)를 사용한 **코드** 구성 방법을 제공합니다. 또한, [IAM을 위한 고급 설정](./iam/advanced_env_vars.md)을 참고하십시오.

## 코드로서의 설정

| 환경 변수                         | 설명                                                                                                                                                                |
|----------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| LICENSE                          | 당신의 wandb/local 라이센스                                                                                                                                         |
| MYSQL                            | MySQL 연결 문자열                                                                                                                                                    |
| BUCKET                           | 데이터를 저장하기 위한 S3 / GCS 버킷                                                                                                                                |
| BUCKET_QUEUE                     | 오브젝트 생성 이벤트를 위한 SQS / Google PubSub 큐                                                                                                                  |
| NOTIFICATIONS_QUEUE              | run 이벤트를 게시할 SQS 큐                                                                                                                                           |
| AWS_REGION                       | 버킷이 위치한 AWS 지역                                                                                                                                                |
| HOST                             | 인스턴스의 FQD, 예: [https://my.domain.net](https://my.domain.net)                                                                                                   |
| OIDC_ISSUER                      | Open ID Connect ID 공급자에 대한 URL, 예: [https://cognito-idp.us-east-1.amazonaws.com/us-east-1_uiIFNdacd](https://cognito-idp.us-east-1.amazonaws.com/us-east-1_uiIFNdacd) |
| OIDC_CLIENT_ID                   | ID 공급자에서 애플리케이션의 클라이언트 ID                                                                                                                            |
| OIDC_AUTH_METHOD                 | 암시적(기본값) 혹은 pkce, 자세한 내용은 아래를 참고하십시오                                                                                                          |
| SLACK_CLIENT_ID                  | 알림을 위해 사용할 Slack 애플리케이션의 클라이언트 ID                                                                                                               |
| SLACK_SECRET                     | 알림을 위해 사용할 Slack 애플리케이션의 비밀                                                                                                                                           |
| LOCAL_RESTORE                    | 인스턴스를 엑세스할 수 없는 경우 일시적으로 true로 설정할 수 있습니다. 컨테이너의 로그에서 임시 자격 증명을 확인하십시오.                                              |
| REDIS                            | 외부 REDIS 인스턴스를 W&B와 함께 설정할 수 있습니다.                                                                                                                 |
| LOGGING_ENABLED                  | true로 설정하면 엑세스 로그가 stdout으로 스트리밍됩니다. 이 변수를 설정하지 않고도 사이드카 컨테이너를 마운트하여 `/var/log/gorilla.log`를 감시할 수 있습니다.          |
| GORILLA_ALLOW_USER_TEAM_CREATION | true로 설정하면 관리자가 아닌 사용자에게 새 팀을 만들 수 있는 권한을 줍니다. 기본값은 False입니다.                                                                          |
| GORILLA_DATA_RETENTION_PERIOD    | 삭제된 run 데이터의 보존 기간을 시간 단위로 설정합니다. 삭제된 run 데이터는 복구할 수 없습니다. 입력 값에 `h`를 추가하십시오. 예를 들어, `"24h"`처럼 설정합니다.  |
| ENABLE_REGISTRY_UI               | true로 설정하면 새로운 W&B 등록 UI가 활성화됩니다.                                                                                                                     |

:::안내

GORILLA_DATA_RETENTION_PERIOD 환경 변수를 신중히 사용하십시오. 환경 변수를 설정하면 데이터는 즉시 제거됩니다. 이 플래그를 활성화하기 전에 데이터베이스와 스토리지 버킷을 백업하는 것도 권장합니다.

:::

## 고급 신뢰성 설정

#### Redis

외부 redis 서버를 구성하는 것은 선택 사항이지만, 프로덕션 시스템에서는 강력히 추천됩니다. Redis는 서비스의 신뢰성을 향상시키고 캐싱을 활성화하여 특히 대형 프로젝트에서 로드 시간을 줄입니다. 높은 가용성(HA)을 갖춘 관리형 redis 서비스(예: ElastiCache)와 다음 사양을 사용하는 것을 권장합니다:

- 최소 4GB 메모리, 8GB 권장
- Redis 버전 6.x
- 전송 중 암호화
- 인증 활성화

#### W&B 서버에서 REDIS 구성

W&B에서 redis 인스턴스를 구성하려면, `http(s)://YOUR-W&B-SERVER-HOST/system-admin`의 W&B 설정 페이지로 이동하십시오. "외부 Redis 인스턴스 사용" 옵션을 활성화하고 다음 형식의 `redis` 연결 문자열을 입력하십시오:

![Configuring REDIS in W&B](/images/hosting/configure_redis.png)

환경 변수 `REDIS`를 사용하여 컨테이너나 Kubernetes 배포에서 `redis`를 구성할 수도 있습니다. 또한, `REDIS`를 Kubernetes 비밀로 설정할 수도 있습니다.

위의 경우는 `redis` 인스턴스가 기본 포트 `6379`에서 실행되고 있다고 가정합니다. 다른 포트를 구성하고 인증을 설정하며 `redis` 인스턴스에 TLS를 활성화하려는 경우, 연결 문자열 형식은 다음과 같습니다: `redis://$USER:$PASSWORD@$HOST:$PORT?tls=true`
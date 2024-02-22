---
description: How to configure the W&B Server installation
displayed_sidebar: default
---

# 환경 변수

인스턴스 레벨 설정을 시스템 설정 관리 UI를 통해 구성하는 것 외에도 W&B는 환경 변수를 사용하여 코드를 통해 이러한 값을 구성할 수 있는 방법을 제공합니다.

## 코드로 구성하기

| 환경 변수                         | 설명                                                                                                                                                                              |
|----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| LICENSE                          | 당신의 wandb/local 라이선스                                                                                                                                                                 |
| MYSQL                            | MySQL 연결 문자열                                                                                                                                                              |
| BUCKET                           | 데이터 저장을 위한 S3 / GCS 버킷                                                                                                                                                     |
| BUCKET_QUEUE                     | 개체 생성 이벤트를 위한 SQS / Google PubSub 큐                                                                                                                                 |
| NOTIFICATIONS_QUEUE              | 실행 이벤트를 게시할 SQS 큐                                                                                                                                             |
| AWS_REGION                       | 버킷이 위치한 AWS 리전                                                                                                                                                   |
| HOST                             | 인스턴스의 FQD, 예: [https://my.domain.net](https://my.domain.net)                                                                                                            |
| OIDC_ISSUER                      | Open ID Connect 신원 제공자의 URL, 예: [https://cognito-idp.us-east-1.amazonaws.com/us-east-1_uiIFNdacd](https://cognito-idp.us-east-1.amazonaws.com/us-east-1_uiIFNdacd) |
| OIDC_CLIENT_ID                   | 신원 제공자의 애플리케이션 클라이언트 ID                                                                                                                                   |
| OIDC_AUTH_METHOD                 | 암시적(기본값) 또는 pkce, 더 많은 맥락은 아래 참조                                                                                                                                   |
| SLACK_CLIENT_ID                  | 알림용으로 사용하려는 Slack 애플리케이션의 클라이언트 ID                                                                                                                        |
| SLACK_SECRET                     | 알림용으로 사용하려는 Slack 애플리케이션의 비밀                                                                                                                           |
| LOCAL_RESTORE                    | 인스턴스에 엑세스할 수 없는 경우 이를 임시적으로 true로 설정하십시오. 컨테이너의 로그에서 임시 자격증명을 확인하십시오.                                              |
| REDIS                            | W&B와 함께 외부 REDIS 인스턴스를 설정하는데 사용할 수 있습니다.                                                                                                                                |
| LOGGING_ENABLED                  | true로 설정하면 엑세스 로그가 stdout으로 스트리밍됩니다. 이 변수를 설정하지 않고 사이드카 컨테이너를 마운트하고 `/var/log/gorilla.log`를 테일링할 수도 있습니다.                              |
| GORILLA_ALLOW_USER_TEAM_CREATION | true로 설정하면 비관리자 사용자가 새 팀을 생성할 수 있습니다. 기본적으로 false입니다.                                                                                                         |
| GORILLA_DATA_RETENTION_PERIOD | 실행에서 삭제된 데이터를 몇 시간 동안 유지할지 설정합니다. 삭제된 실행 데이터는 복구할 수 없습니다. 입력값에 `h`를 추가합니다. 예: `"24h"`. |


:::안내

GORILLA_DATA_RETENTION_PERIOD 환경 변수를 신중하게 사용하십시오. 환경 변수가 설정되면 데이터는 즉시 삭제됩니다. 이 플래그를 활성화하기 전에 데이터베이스와 스토리지 버킷을 모두 백업하는 것이 좋습니다.

:::

## 고급 신뢰성 설정

#### Redis

외부 redis 서버를 구성하는 것은 선택 사항이지만, 프로덕션 시스템에서는 강력히 권장됩니다. Redis는 서비스의 신뢰성을 향상시키고 캐싱을 활성화하여 특히 큰 프로젝트에서 로드 시간을 줄여줍니다. 고가용성(HA)과 다음 사양을 갖춘 관리형 redis 서비스(예: ElastiCache) 사용을 권장합니다:

- 최소 4GB의 메모리, 권장 8GB
- Redis 버전 6.x
- 이동 중 암호화
- 인증 활성화

#### W&B 서버에서 REDIS 구성하기

W&B와 함께 redis 인스턴스를 구성하려면 `http(s)://YOUR-W&B-SERVER-HOST/system-admin`에서 W&B 설정 페이지로 이동합니다. "외부 Redis 인스턴스 사용" 옵션을 활성화하고, 다음 형식으로 `redis` 연결 문자열을 입력합니다:

![W&B에서 REDIS 구성하기](/images/hosting/configure_redis.png)

`REDIS` 환경 변수를 사용하여 컨테이너 또는 Kubernetes 배포에서 `redis`를 구성할 수도 있습니다. 또는 `REDIS`를 Kubernetes 비밀로 설정할 수도 있습니다.

위에서는 `redis` 인스턴스가 기본 포트인 `6379`에서 실행되고 있다고 가정합니다. 다른 포트를 구성하고 인증을 설정하며 `redis` 인스턴스에 TLS를 활성화하려는 경우 연결 문자열 형식은 다음과 같습니다: `redis://$USER:$PASSWORD@$HOST:$PORT?tls=true`
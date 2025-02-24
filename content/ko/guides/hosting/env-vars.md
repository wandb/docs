---
title: Configure environment variables
description: W&B 서버 설치를 구성하는 방법
menu:
  default:
    identifier: ko-guides-hosting-env-vars
    parent: w-b-platform
weight: 7
---

System Settings 관리자 UI를 통해 인스턴스 수준 설정을 구성하는 것 외에도, W&B는 환경 변수를 사용하여 코드를 통해 이러한 값을 구성하는 방법도 제공합니다. 또한 [IAM 고급 설정]({{< relref path="./iam/advanced_env_vars.md" lang="ko" >}})을 참조하십시오.

## 환경 변수 참조

| 환경 변수                       | 설명                                                                                                                                                                                 |
|-----------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| LICENSE                           | wandb/local 라이선스                                                                                                                                                                 |
| MYSQL                             | MySQL 연결 문자열                                                                                                                                                                  |
| BUCKET                            | 데이터 저장용 S3 / GCS 버킷                                                                                                                                                         |
| BUCKET_QUEUE                      | 오브젝트 생성 이벤트를 위한 SQS / Google PubSub 대기열                                                                                                                               |
| NOTIFICATIONS_QUEUE               | run 이벤트를 게시할 SQS 대기열                                                                                                                                                        |
| AWS_REGION                        | 버킷이 있는 AWS 리전                                                                                                                                                                |
| HOST                              | 인스턴스의 FQD, 즉 `https://my.domain.net`                                                                                                                                         |
| OIDC_ISSUER                       | Open ID Connect ID 공급자의 URL, 즉 `https://cognito-idp.us-east-1.amazonaws.com/us-east-1_uiIFNdacd`                                                                              |
| OIDC_CLIENT_ID                    | ID 공급자의 애플리케이션의 클라이언트 ID                                                                                                                                               |
| OIDC_AUTH_METHOD                  | Implicit (기본값) 또는 pkce, 자세한 내용은 아래 참조                                                                                                                                      |
| SLACK_CLIENT_ID                   | 알림에 사용할 Slack 애플리케이션의 클라이언트 ID                                                                                                                                        |
| SLACK_SECRET                      | 알림에 사용할 Slack 애플리케이션의 secret                                                                                                                                           |
| LOCAL_RESTORE                     | 인스턴스에 엑세스할 수 없는 경우 임시로 true로 설정할 수 있습니다. 임시 자격 증명에 대한 컨테이너의 로그를 확인하십시오.                                                                      |
| REDIS                             | W&B와 함께 외부 REDIS 인스턴스를 설정하는 데 사용할 수 있습니다.                                                                                                                                |
| LOGGING_ENABLED                   | true로 설정하면 엑세스 로그가 stdout으로 스트리밍됩니다. 사이드카 컨테이너를 마운트하고 이 변수를 설정하지 않고도 `/var/log/gorilla.log`를 테일할 수도 있습니다.                                    |
| GORILLA_ALLOW_USER_TEAM_CREATION | true로 설정하면 비 관리자 **Users**가 새 **Teams**을 만들 수 있습니다. 기본값은 false입니다.                                                                                                        |
| GORILLA_DATA_RETENTION_PERIOD  | 삭제된 run의 데이터를 보관하는 기간(시간)입니다. 삭제된 run 데이터는 복구할 수 없습니다. 입력 값에 `h`를 추가합니다. 예: `"24h"`. |
| ENABLE_REGISTRY_UI                | true로 설정하면 새로운 W&B Registry UI가 활성화됩니다.                                                                                                                               |

{{% alert %}}
GORILLA_DATA_RETENTION_PERIOD 환경 변수를 신중하게 사용하십시오. 환경 변수가 설정되면 데이터가 즉시 제거됩니다. 이 플래그를 활성화하기 전에 데이터베이스와 스토리지 버킷을 모두 백업하는 것이 좋습니다.
{{% /alert %}}

## 고급 안정성 설정

### Redis

외부 Redis 서버 구성은 선택 사항이지만 프로덕션 시스템에 권장됩니다. Redis는 서비스의 안정성을 향상시키고 특히 대규모 **Projects**에서 로드 시간을 줄이기 위해 캐싱을 활성화하는 데 도움이 됩니다. 고가용성(HA) 기능이 있는 ElastiCache와 같은 관리형 Redis 서비스를 사용하고 다음 사양을 따르십시오.

- 최소 4GB 메모리, 8GB 권장
- Redis 버전 6.x
- 전송 중 암호화
- 인증 활성화

W&B로 Redis 인스턴스를 구성하려면 `http(s)://YOUR-W&B-SERVER-HOST/system-admin`에서 W&B 설정 페이지로 이동하십시오. "외부 Redis 인스턴스 사용" 옵션을 활성화하고 다음 형식으로 Redis 연결 문자열을 입력하십시오.

{{< img src="/images/hosting/configure_redis.png" alt="W&B에서 REDIS 구성" >}}

컨테이너 또는 Kubernetes 배포에서 환경 변수 `REDIS`를 사용하여 Redis를 구성할 수도 있습니다. 또는 `REDIS`를 Kubernetes secret으로 설정할 수도 있습니다.

이 페이지에서는 Redis 인스턴스가 기본 포트인 `6379`에서 실행 중이라고 가정합니다. 다른 포트를 구성하고 인증을 설정하고 `redis` 인스턴스에서 TLS를 활성화하려면 연결 문자열 형식이 `redis://$USER:$PASSWORD@$HOST:$PORT?tls=true`와 같이 표시됩니다.

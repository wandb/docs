---
slug: /guides/hosting
displayed_sidebar: default
---

# W&B 서버

W&B를 W&B가 관리하거나 직접 관리하는 리소스 격리 환경에서 배포하세요. W&B 서버는 어떤 기반 인프라에도 쉽게 배포할 수 있는 패키지된 Docker 이미지로 제공됩니다. 다양한 환경에서 W&B 서버를 설치하고 호스트하는 여러 방법이 있습니다.

:::info
W&B 서버의 프로덕션급 기능은 엔터프라이즈 등급에서만 사용할 수 있습니다.

개발자 또는 시행 환경을 설정하는 방법은 [기본 설정 가이드](./how-to-guides/basic-setup.md)를 참조하세요.
:::

W&B 서버를 사용하면 다음과 같은 기능을 구성하고 활용할 수 있습니다:

- [안전한 저장소 커넥터](./secure-storage-connector.md)
- [싱글 사인온](./sso.md)
- [LDAP를 통한 역할 기반 엑세스 제어](./ldap.md)
- [감사 로그](./audit-logging.md)
- [사용자 관리](./manage-users.md)
- [프로메테우스 모니터링](./prometheus-logging.md)
- [슬랙 알림](./slack-alerts.md)
- [삭제된 아티팩트 수거](../artifacts/delete-artifacts.md#how-to-enable-garbage-collection-based-on-how-you-host-wb)

다음 문서 섹션들은 W&B 서버를 설치하는 다양한 옵션, 공유 책임 모델, 단계별 설치 및 구성 가이드를 설명합니다.

## 권장 사항

W&B 서버를 구성할 때 다음을 권장합니다:

1. 외부 저장소와 외부 MySQL 데이터베이스를 사용하여 W&B 서버 Docker 컨테이너를 실행해 컨테이너가 종료되거나 충돌할 경우 데이터가 우연히 삭제되는 것을 방지합니다. 이는 데이터가 컨테이너의 다른 버전 간에 보존될 수 있도록 보호합니다.
2. `wandb` 서비스를 노출하기 위해 쿠버네티스를 활용하여 W&B 서버 Docker 이미지를 실행합니다.
3. 프로덕션 관련 작업에 W&B 서버를 사용할 계획이라면 확장 가능한 파일 시스템을 설정하고 관리하세요.

## 시스템 요구 사항

W&B 서버는 적어도

- 4개의 CPU 코어 &
- 8GB의 메모리(RAM)

를 갖춘 기계가 필요합니다.

귀하의 W&B 데이터는 영구 볼륨 또는 외부 데이터베이스에 저장되어 컨테이너의 다른 버전 간에 보존됩니다.

:::tip
엔터프라이즈 고객을 위해 W&B는 개인 호스팅 인스턴스에 대해 광범위한 기술 지원과 자주 있는 설치 업데이트를 제공합니다.
:::

## 릴리스

새로운 W&B 서버 릴리스가 나올 때 알림을 받으려면 [W&B 서버 GitHub 저장소](https://github.com/wandb/server/releases)를 구독하세요.

구독하려면 GitHub 페이지 상단의 **Watch** 버튼을 선택하고 **All Activity**를 선택합니다.
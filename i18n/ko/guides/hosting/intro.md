---
slug: /guides/hosting
displayed_sidebar: default
---

# W&B 서버

W&B를 W&B 또는 직접 관리하는 자원 격리 환경에서 배포하세요. W&B 서버는 모든 기반 인프라에 쉽게 배포할 수 있는 패키지화된 Docker 이미지로 제공됩니다. W&B 서버를 다양한 환경에서 설치하고 호스팅하는 여러 가지 방법이 있습니다.

:::info
W&B 서버의 프로덕션 등급 기능은 엔터프라이즈 계층만 사용할 수 있습니다.

개발자 또는 트라이얼 환경을 설정하는 [기본 설정 가이드](./how-to-guides/basic-setup.md)를 참조하세요.
:::

W&B 서버를 사용하면 다음과 같은 기능을 구성하고 활용할 수 있습니다:

- [안전한 저장소 커넥터](./secure-storage-connector.md)
- [싱글 사인온](./sso.md)
- [LDAP를 통한 역할 기반 엑세스 제어](./ldap.md)
- [감사 로그](./audit-logging.md)
- [사용자 관리](./manage-users.md)
- [프로메테우스 모니터링](./prometheus-logging.md)
- [슬랙 알림](./slack-alerts.md)
- [삭제된 아티팩트의 쓰레기 수집](../artifacts/delete-artifacts.md#how-to-enable-garbage-collection-based-on-how-you-host-wb)

이 문서의 다음 섹션에서는 W&B 서버를 설치하는 다양한 옵션, 공동 책임 모델, 단계별 설치 및 설정 가이드를 설명합니다.

## 권장 사항

W&B 서버를 구성할 때 W&B는 다음을 권장합니다:

1. 컨테이너 외부에 상태를 보존하기 위해 외부 저장소와 외부 MySQL 데이터베이스를 사용하여 W&B 서버 Docker 컨테이너를 실행하세요. 이렇게 하면 컨테이너가 중단되거나 충돌하는 경우 데이터가 실수로 삭제되는 것을 방지할 수 있습니다.
2. `wandb` 서비스를 노출시키기 위해 쿠버네티스를 활용하여 W&B 서버 Docker 이미지를 실행하세요.
3. 프로덕션 관련 작업을 위해 W&B 서버를 사용할 계획이라면 확장 가능한 파일 시스템을 설정하고 관리하세요.

## 시스템 요구 사항

W&B 서버는 최소한

- CPU 4 코어 &
- 메모리(RAM) 8GB

을 요구하는 기계가 필요합니다.

W&B 데이터는 지속적인 볼륨 또는 외부 데이터베이스에 저장되므로 컨테이너의 다른 버전 간에도 보존됩니다.

:::tip
엔터프라이즈 고객을 위해, W&B는 사설 호스팅 인스턴스에 대해 광범위한 기술 지원 및 자주 업데이트되는 설치를 제공합니다.
:::

## 릴리스

새로운 W&B 서버 릴리스가 나올 때 [W&B 서버 GitHub 저장소](https://github.com/wandb/server/releases)에서 알림을 받으세요.

구독하려면 GitHub 페이지 상단의 **Watch** 버튼을 선택하고 **All Activity**를 선택하세요.
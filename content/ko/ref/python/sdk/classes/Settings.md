---
title: 설정
data_type_classification: class
menu:
  reference:
    identifier: ko-ref-python-sdk-classes-Settings
object_type: python_sdk_actions
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_settings.py >}}

W&B SDK에 대한 설정입니다.

이 클래스는 W&B SDK의 설정을 관리하며, 모든 설정의 타입 안전성과 검증을 보장합니다. 설정들은 속성(attribute) 형태로 엑세스할 수 있으며, 프로그래밍적으로 또는 환경 변수(`WANDB_` 접두사 사용), 그리고 설정 파일을 통해 초기화할 수 있습니다.

설정은 세 가지 카테고리로 구성되어 있습니다:
1. 공개 설정: 사용자가 자신의 필요에 맞게 W&B의 동작을 안전하게 커스터마이즈할 수 있는 핵심 설정 옵션입니다.
2. 내부 설정: 'x_'로 시작하는 이름의 설정으로, 낮은 수준의 SDK 동작을 제어합니다. 주로 내부용 및 디버깅을 위해 사용됩니다. 수정은 가능하지만, 공개 API에는 포함되지 않아 향후 버전에서 예고 없이 변경될 수 있습니다.
3. 계산된 설정: 다른 설정이나 환경으로부터 자동으로 도출되는 읽기 전용 설정입니다.

속성(Attributes):

- allow_offline_artifacts (bool): offline 모드에서 table Artifacts를 동기화할 수 있도록 허용하는 플래그입니다. 이전 동작으로 되돌리고 싶다면 False로 설정하세요.
- allow_val_change (bool): `Config` 값이 한 번 설정된 이후에도 수정할 수 있도록 허용하는 플래그입니다.
- anonymous (Optional): 익명 데이터 로깅 방식을 설정합니다.
    가능한 값:
    - "never": run을 추적하기 전에 반드시 W&B 계정을 연결해야 하며, 실수로 익명 run을 생성하지 않게 합니다.
    - "allow": 로그인한 사용자는 계정으로 run을 추적할 수 있고, W&B 계정 없이 스크립트를 실행하는 사람은 UI에서 차트를 볼 수 있습니다.
    - "must": run을 등록된 사용자 계정 대신 익명 계정으로 전송합니다.
- api_key (Optional): W&B API 키.
- azure_account_url_to_access_key (Optional): Azure 인테그레이션을 위한 Azure 계정 URL과 그에 해당하는 엑세스 키 매핑.
- base_url (str): 데이터 동기화에 사용하는 W&B 백엔드의 URL.
- code_dir (Optional): W&B가 추적할 코드를 포함한 디렉토리.
- config_paths (Optional): 설정 파일의 경로 리스트로, 해당 파일에서 `Config` 오브젝트에 설정을 로드합니다.
- console (Literal): 적용할 콘솔 캡처 타입.
    가능한 값:
    "auto" - 시스템 환경과 설정에 따라 콘솔 캡처 방식을 자동으로 선택합니다.
    "off" - 콘솔 캡처를 비활성화합니다.
    "redirect" - 출력 캡처를 위해 하위 파일 디스크립터를 리디렉션합니다.
    "wrap" - sys.stdout/sys.stderr의 write 메소드를 재정의합니다. 시스템 상태에 따라 "wrap_raw" 또는 "wrap_emu"로 매핑됩니다.
    "wrap_raw" - "wrap"과 같지만 에뮬레이터 없이 원시 출력을 직접 캡처합니다. `wrap` 설정에서 파생되므로 수동 설정은 권장하지 않습니다.
    "wrap_emu" - "wrap"과 같지만 에뮬레이터를 통해 출력을 캡처합니다. `wrap` 설정에서 파생되므로 수동 설정은 권장하지 않습니다.
- console_multipart (bool): 멀티파트 콘솔 로그 파일을 생성할지 여부.
- credentials_file (str): 임시 엑세스 토큰을 저장할 파일의 경로.
- disable_code (bool): 코드 캡처를 비활성화할지 여부.
- disable_git (bool): git 상태 캡처를 비활성화할지 여부.
- disable_job_creation (bool): W&B Launch의 job artifact 생성을 비활성화할지 여부.
- docker (Optional): 스크립트 실행에 사용하는 Docker 이미지.
- email (Optional): 사용자의 이메일 어드레스.
- entity (Optional): W&B의 entity, 예를 들어 사용자 또는 팀.
- force (bool): `wandb.login()`에 `force` 플래그를 전달할지 여부.
- fork_from (Optional): 이전 run 실행의 특정 지점(run ID, 메트릭, 그 값)에서 포크할 지점 지정. 현재는 '_step' 메트릭만 지원합니다.
- git_commit (Optional): run에 연결할 git 커밋 해시.
- git_remote (str): run에 연결할 git remote.
- git_remote_url (Optional): git remote 저장소의 URL.
- git_root (Optional): git 저장소의 루트 디렉토리.

- host (Optional): 스크립트가 실행 중인 머신의 호스트명.
- http_proxy (Optional): W&B로의 http 요청에 사용할 커스텀 프록시 서버.
- https_proxy (Optional): W&B로의 https 요청에 사용할 커스텀 프록시 서버.
- identity_token_file (Optional): 인증을 위한 identity 토큰(JWT)이 포함된 파일 경로.
- ignore_globs (Sequence): 업로드에서 제외할 파일을 `files_dir` 기준 유닉스 glob 패턴으로 지정합니다.
- init_timeout (float): `wandb.init` 호출이 완료될 때까지 대기할 시간(초).
- insecure_disable_ssl (bool): SSL 검증을 비활성화할지 여부(주의 필요).
- job_name (Optional): 현재 실행 중인 Launch job의 이름.
- job_source (Optional): Launch의 소스 타입.
- label_disable (bool): 자동 라벨링 기능 비활성화 여부.

- launch_config_path (Optional): launch 설정 파일 경로.
- login_timeout (Optional): 로그인 작업이 완료될 때까지 대기할 시간(초) 설정.
- mode (Literal): W&B 로깅 및 동기화용 운영 모드.
- notebook_name (Optional): 주피터 같은 노트북 환경에서 실행할 경우 노트북의 이름.
- organization (Optional): W&B 조직.
- program (Optional): run을 생성한 스크립트의 경로(가능한 경우).
- program_abspath (Optional): run을 생성한 스크립트의 루트 저장소 기준 절대 경로. 루트 저장소 디렉토리는 .git 폴더가 있으면 해당 위치, 없으면 현재 작업 디렉토리.
- program_relpath (Optional): run을 생성한 스크립트의 상대경로.
- project (Optional): W&B 프로젝트 ID.
- quiet (bool): 필수적이지 않은 출력 억제 여부.
- reinit (Union): 이미 활성화된 run 중에 `wandb.init()` 호출 시 어떻게 할지 선택.
    옵션:
    - "default": 노트북에서는 "finish_previous", 그 외에는 "return_previous"를 사용.
    - "return_previous": 아직 종료되지 않은 최신 run을 반환(단, `wandb.run`은 업데이트되지 않음).
    - "finish_previous": 모든 활성 run을 종료한 후 새 run 반환.
    - "create_new": 다른 활성 run에 영향 없이 새 run 생성(`wandb.run` 및 `wandb.log` 등 최상위 함수 미반영). 오래된 일부 인테그레이션에서는 미지원.
    불리언 값도 가능하나 deprecated. False는 "return_previous", True는 "finish_previous"와 동일.
- relogin (bool): 새로운 로그인 시도를 강제할지 여부.
- resume (Optional): run의 resume 동작 지정.
    옵션:
    - "must": 동일 ID의 기존 run에서만 resume. 없으면 실패 처리.
    - "allow": 동일 ID의 기존 run에서 resume 시도. 없으면 새 run 생성.
    - "never": 항상 새 run 시작. 동일 ID의 run이 있으면 실패 처리.
    - "auto": 동일 머신에서 최근 실패한 run 자동 resume.
- resume_from (Optional): 이전 run 실행의 특정 지점(run ID, 메트릭, 그 값)을 resume할 위치로 지정. 현재는 '_step' 메트릭만 지원.

- root_dir (str): 모든 run 관련 경로의 기준이 되는 루트 디렉토리. 특히 wandb 디렉토리와 run 디렉토리를 만드는데 사용됨.
- run_group (Optional): 관련된 run들을 식별하는 그룹 ID. UI에서 run들을 그룹화할 때 사용.
- run_id (Optional): run의 ID.
- run_job_type (Optional): 실행되는 job의 타입(예: training, evaluation).
- run_name (Optional): 사람이 알아보기 쉬운 run의 이름.
- run_notes (Optional): run에 추가할 노트 또는 설명.
- run_tags (Optional): run에 태그를 지정해 구성 및 필터링에 활용.
- sagemaker_disable (bool): SageMaker 전용 기능 비활성화 플래그.
- save_code (Optional): run에 연결된 코드 저장 여부.
- settings_system (Optional): 시스템 전체에서 사용하는 설정 파일 경로.

- show_errors (bool): 에러 메시지 출력 여부.
- show_info (bool): 안내 메시지 출력 여부.
- show_warnings (bool): 경고 메시지 출력 여부.
- silent (bool): 모든 출력 억제 플래그.

- strict (Optional): 검증 및 에러 체크 강제 모드 활성화 여부.
- summary_timeout (int): summary 작업이 완료될 때까지 대기할 시간(초).

- sweep_id (Optional): 이 run이 속한 sweep의 식별자.
- sweep_param_path (Optional): sweep 파라미터 설정 경로.
- symlink (bool): 심볼릭 링크 사용 여부(Windows 제외 기본 True).
- sync_tensorboard (Optional): TensorBoard 로그를 W&B와 동기화할지 여부.
- table_raise_on_max_row_limit_exceeded (bool): table 행(row) 제한 초과 시 예외 발생 여부.
- username (Optional): 사용자명.

- x_skip_transaction_log (bool): run 이벤트를 transaction 로그에 저장하지 않을지 여부. 온라인 run에서만 관련 있음. 디스크에 기록되는 데이터 양을 줄이기 위해 사용 가능. recoverability(복구 보장)을 희생할 수 있으므로 사용 시 주의 필요.

- x_stats_open_metrics_endpoints (Optional): 시스템 메트릭을 모니터링하기 위한 OpenMetrics `/metrics` 엔드포인트.
- x_stats_open_metrics_filters (Union): OpenMetrics `/metrics` 엔드포인트에서 수집한 메트릭에 적용할 필터.
    두 가지 형식 지원:
    - {"엔드포인트 이름을 포함한 메트릭 정규표현식": {"label": "label 값 정규표현식"}}
    - ("메트릭 정규표현식 1", "메트릭 정규표현식 2", ...)
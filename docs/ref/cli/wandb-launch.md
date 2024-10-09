# wandb launch

**사용법**

`wandb launch [옵션]`

**요약**

W&B Job을 실행하거나 대기열에 추가합니다. 자세한 내용은 https://wandb.me/launch를 참조하세요.

**옵션**

| **옵션** | **설명** |
| :--- | :--- |
| -u, --uri (str) | 실행할 로컬 경로 또는 git repo uri입니다. 이 옵션을 제공하면 지정된 uri에서 job을 생성합니다. |
| -j, --job (str) | 실행할 job의 이름입니다. 이 옵션을 사용하면 실행 시 uri가 필요하지 않습니다. |
| --entry-point | Project 내의 entry point입니다. [기본값: main]. entry point가 없으면 지정된 이름의 프로젝트 파일을 실행하려고 시도합니다. '.py' 파일은 'python'으로 실행하고, '.sh' 파일은 환경 변수 $SHELL에 의해 지정된 기본 셸로 실행됩니다. 이 옵션을 사용하면 설정 파일을 통해 전달된 entrypoint 값을 덮어씁니다. |
| --build-context (str) | 소스 코드 내의 빌드 컨텍스트 경로입니다. 기본값은 소스 코드의 루트입니다. -u와 함께 사용할 수 있습니다. |
| --name | run을 실행할 run의 이름입니다. 지정하지 않으면 실행을 위한 임의의 run 이름이 사용됩니다. 이 옵션을 사용하면 설정 파일을 통해 전달된 이름을 덮어씁니다. |
| -e, --entity (str) | 새 run이 전송될 대상 엔터티의 이름입니다. 기본값은 로컬 wandb/settings 폴더에 설정된 엔터티를 사용합니다. 이 옵션을 사용하면 설정 파일을 통해 전달된 엔터티 값을 덮어씁니다. |
| -p, --project (str) | 새 run이 전송될 대상 프로젝트의 이름입니다. 기본값은 소스 uri 또는 github run의 경우 git repo 이름에서 제공된 프로젝트 이름을 사용합니다. 이 옵션을 사용하면 설정 파일을 통해 전달된 프로젝트 값을 덮어씁니다. |
| -r, --resource | run에 사용할 실행 리소스입니다. 지원되는 값: 'local-process', 'local-container', 'kubernetes', 'sagemaker', 'gcp-vertex'. 이 값은 리소스 설정 없이 대기열에 푸시할 경우 이제 필수 파라미터입니다. 이 옵션을 사용하면 설정 파일을 통해 전달된 리소스 값을 덮어씁니다. |
| -d, --docker-image | 사용하고 싶은 특정 Docker 이미지입니다. 형태는 name:tag입니다. 이 옵션을 사용하면 설정 파일을 통해 전달된 Docker 이미지 값을 덮어씁니다. |
| --base-image | job 코드를 실행할 Docker 이미지입니다. --docker-image와 호환되지 않습니다. |
| -c, --config | JSON 파일 경로(확장자가 '.json'이어야 함) 또는 JSON 문자열로 런치 설정으로 전달됩니다. 실행된 run이 어떻게 설정될지 결정합니다. |
| -v, --set-var | 키-값 쌍으로 큐의 허용 목록에서 템플릿 변수 값을 설정합니다. 예: `--set-var key1=value1 --set-var key2=value2` |
| -q, --queue | 푸시할 run 큐의 이름입니다. 지정하지 않으면 단일 run을 직접 실행합니다. 인수 없이 제공되면(`--queue`), 기본적으로 'default' 큐로 설정합니다. 그렇지 않고 이름이 제공되면, 지정된 프로젝트 및 엔터티 아래에 해당 run 큐가 존재해야 합니다. |
| --async | Job을 비동기적으로 실행하는 플래그입니다. 기본값은 false입니다. 즉, --async가 설정되지 않으면 wandb launch는 job이 완료될 때까지 기다립니다. 이 옵션은 --queue와 호환되지 않습니다; 에이전트와 함께 실행할 때 비동기 옵션은 wandb launch-agent에서 설정해야 합니다. |
| --resource-args | JSON 파일 경로(확장자가 '.json'이어야 함) 또는 JSON 문자열로서 컴퓨트 리소스에 전달될 리소스 인수입니다. 제공되어야 할 정확한 내용은 각 실행 백엔드마다 다릅니다. 이 파일의 레이아웃에 대한 문서를 참조하세요. |
| --dockerfile | job을 빌드하는 데 사용된 Dockerfile의 경로입니다. job의 루트에서 상대적입니다. |
| --priority [critical|high|medium|low] | --queue가 전달되었을 때, job의 우선 순위를 설정합니다. 높은 우선 순위를 가진 런치 job이 먼저 서비스됩니다. 우선 순위가 높은 순서부터 낮은 순서대로: critical, high, medium, low |
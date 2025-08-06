---
title: 실험 재현
menu:
  default:
    identifier: ko-guides-models-track-reproduce_experiments
    parent: track
weight: 7
---

재현을 통해 팀 멤버가 생성한 실험의 결과를 검증하고 확인하세요.

실험을 재현하기 전에 아래 정보를 확인해 두세요:

* run 이 기록된 Project 이름
* 재현하려는 run 이름

실험을 재현하려면 다음 순서를 따르세요:

1. run 이 기록된 Project로 이동합니다.
2. 왼쪽 사이드바에서 **Workspace** 탭을 선택합니다.
3. run 목록에서 재현하려는 run 을 선택합니다.
4. **Overview**를 클릭합니다.

계속하려면 주어진 해시를 통해 실험의 코드 를 다운로드하거나, 실험의 전체 저장소를 클론하세요.

{{< tabpane text=true >}}
{{% tab "Python 스크립트 또는 노트북 다운로드" %}}

실험의 Python 스크립트 또는 노트북을 다운로드하려면:

1. **Command** 필드에서 실험을 생성한 스크립트의 이름을 메모해 두세요.
2. 왼쪽 네비게이션 바에서 **Code** 탭을 선택합니다.
3. 해당 스크립트나 노트북에 해당하는 파일 옆의 **Download**를 클릭합니다.

{{% /tab %}}
{{% tab "GitHub" %}}

팀원이 실험을 만들 때 사용한 GitHub 저장소를 클론하세요. 방법은 다음과 같습니다:

1. 필요하다면, 팀원이 실험에 사용한 GitHub 저장소에 접근 권한을 얻으세요.
2. GitHub 저장소 URL이 들어있는 **Git repository** 필드를 복사합니다.
3. 저장소를 클론합니다:
    ```bash
    git clone https://github.com/your-repo.git && cd your-repo
    ```
4. **Git state** 필드를 복사해서 터미널에 붙여넣으세요. Git state는 실험 생성 시 사용된 정확한 커밋으로 체크아웃하는 Git 명령어 집합입니다. 아래 코드 내 값을 본인 환경에 맞게 수정하세요:
    ```bash
    git checkout -b "<run-name>" 0123456789012345678901234567890123456789
    ```

{{% /tab %}}
{{< /tabpane >}}

5. 왼쪽 네비게이션 바에서 **Files**를 선택합니다.
6. `requirements.txt` 파일을 다운로드해 작업 디렉토리에 저장하세요. 이 디렉토리에는 클론한 GitHub 저장소 또는 다운로드한 Python 스크립트나 노트북이 위치해야 합니다.
7. (권장) Python 가상 환경을 생성하세요.
8. `requirements.txt` 파일에 지정된 패키지를 설치합니다.
    ```bash
    pip install -r requirements.txt
    ```

9. 이제 코드와 의존성 설치가 완료되었으니, 스크립트나 노트북을 실행해 실험을 재현할 수 있습니다. 저장소를 클론했다면, 스크립트나 노트북이 위치한 디렉토리로 이동해야 할 수 있습니다. 그렇지 않으면, 바로 작업 디렉토리에서 실행하셔도 됩니다.

{{< tabpane text=true >}}
{{% tab "Python 노트북" %}}

Python 노트북을 다운로드했다면, 해당 디렉토리로 이동한 뒤 아래 명령어를 터미널에 입력하세요:
```bash
jupyter notebook
```

{{% /tab %}}
{{% tab "Python 스크립트" %}}

Python 스크립트를 다운로드했다면, 해당 스크립트가 위치한 디렉토리로 이동한 후 아래 명령어를 터미널에서 입력하세요. `<>`로 표시된 부분은 본인 상황에 맞게 수정하세요:

```bash
python <your-script-name>.py
```

{{% /tab %}}
{{< /tabpane >}}
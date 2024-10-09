# Data Types

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/__init__.py' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

이 모듈은 W&B에 풍부하고 상호작용적인 시각화를 로그하기 위한 데이터 타입을 정의합니다.

데이터 타입에는 이미지, 오디오 및 비디오와 같은 일반 미디어 타입,
테이블 및 HTML과 같은 유연한 정보 컨테이너 등이 포함됩니다.

미디어 로그에 대한 자세한 내용은 [우리의 가이드](https://docs.wandb.com/guides/track/log/media)를 참조하세요.

인터랙티브 데이터셋 및 모델 분석을 위한 구조화된 데이터 로그에 대한 자세한 내용은
[W&B Tables에 대한 가이드](https://docs.wandb.com/guides/data-vis)를 참조하세요.

이러한 특별한 데이터 타입은 모두 WBValue의 서브클래스입니다. 모든 데이터 타입은 JSON으로 직렬화되며, 이는 wandb가 오브젝트를 로컬에 저장하고 W&B 서버로 업로드하는 데 사용하기 때문입니다.

## Classes

[`class Audio`](./audio.md): 오디오 클립을 위한 Wandb 클래스.

[`class BoundingBoxes2D`](./boundingboxes2d.md): 2D 바운딩 박스 오버레이가 있는 이미지를 W&B에 로그하기 위한 형식.

[`class Graph`](./graph.md): 그래프를 위한 Wandb 클래스.

[`class Histogram`](./histogram.md): 히스토그램을 위한 wandb 클래스.

[`class Html`](./html.md): 임의의 HTML을 위한 Wandb 클래스.

[`class Image`](./image.md): W&B에 이미지를 로그하기 위한 형식.

[`class ImageMask`](./imagemask.md): 이미지 마스크 또는 오버레이를 W&B에 로그하기 위한 형식.

[`class Molecule`](./molecule.md): 3D 분자 데이터를 위한 Wandb 클래스.

[`class Object3D`](./object3d.md): 3D 포인트 클라우드를 위한 Wandb 클래스.

[`class Plotly`](./plotly.md): plotly 플롯을 위한 Wandb 클래스.

[`class Table`](./table.md): 표 형식 데이터를 표시하고 분석하는 데 사용되는 Table 클래스.

[`class Video`](./video.md): W&B에 로그하기 위한 비디오 형식.

[`class WBTraceTree`](./wbtracetree.md): 트레이스 트리 데이터를 위한 미디어 오브젝트.
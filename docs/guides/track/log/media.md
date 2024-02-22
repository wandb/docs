---
description: Log rich media, from 3D point clouds and molecules to HTML and histograms
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 미디어 및 개체 로깅

이미지, 비디오, 오디오 등을 지원합니다. 풍부한 미디어를 로깅하여 결과를 탐색하고 실행, 모델 및 데이터세트를 시각적으로 비교하세요. 예시와 사용 방법 가이드를 읽어보세요.

:::info
미디어 유형에 대한 참조 문서를 찾고 계신가요? [이 페이지](../../../ref/python/data-types/README.md)를 참조하세요.
:::

:::info
이러한 미디어 개체를 모두 로깅하는 작동 코드를 [이 Colab 노트북](http://wandb.me/media-colab)에서 확인할 수 있으며, wandb.ai에서 결과를 [여기](https://wandb.ai/lavanyashukla/visualize-predictions/reports/Visualize-Model-Predictions--Vmlldzo1NjM4OA)에서 확인하고, 위에 링크된 비디오 튜토리얼을 따라할 수 있습니다.
:::

## 이미지

입력, 출력, 필터 가중치, 활성화 등을 추적하기 위해 이미지를 로깅하세요!

![인페인팅을 수행하는 오토인코더 네트워크의 입력 및 출력.](/images/track/log_images.png)

이미지는 NumPy 배열, PIL 이미지 또는 파일 시스템에서 직접 로깅할 수 있습니다.

:::info
학습 중 로깅이 병목 현상을 초래하고 결과를 보는 동안 이미지 로딩이 병목 현상이 되지 않도록 단계당 50개 미만의 이미지를 로깅하는 것이 권장됩니다.
:::

<Tabs
  defaultValue="arrays"
  values={[
    {label: '배열을 이미지로 로깅하기', value: 'arrays'},
    {label: 'PIL 이미지 로깅하기', value: 'pil_images'},
    {label: '파일에서 이미지 로깅하기', value: 'images_files'},
  ]}>
  <TabItem value="arrays">

예를 들어, [`torchvision`](https://pytorch.org/vision/stable/utils.html#torchvision.utils.make\_grid)에서 `make_grid`를 사용하여 수동으로 이미지를 구성할 때 배열을 직접 제공합니다.

배열은 [Pillow](https://pillow.readthedocs.io/en/stable/index.html)를 사용하여 png로 변환됩니다.

```python
images = wandb.Image(image_array, caption="Top: Output, Bottom: Input")

wandb.log({"examples": images})
```

마지막 차원이 1이면 이미지가 그레이스케일, 3이면 RGB, 4이면 RGBA라고 가정합니다. 배열에 실수가 포함된 경우, `0`과 `255` 사이의 정수로 변환합니다. 이미지를 다르게 정규화하려면 [`mode`](https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes)를 수동으로 지정하거나, 이 패널의 "PIL 이미지 로깅하기" 탭에서 설명한 대로 [`PIL.Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html)을 직접 제공할 수 있습니다.
  </TabItem>
  <TabItem value="pil_images">

배열에서 이미지로의 변환을 완전히 제어하려면 [`PIL.Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html)를 직접 구성하고 제공하세요.

```python
images = [PIL.Image.fromarray(image) for image in image_array]

wandb.log({"examples": [wandb.Image(image) for image in images]})
```
  </TabItem>
  <TabItem value="images_files">
더 많은 제어를 위해 이미지를 원하는 대로 생성하고, 디스크에 저장한 후 파일 경로를 제공하세요.

```python
im = PIL.fromarray(...)
rgb_im = im.convert("RGB")
rgb_im.save("myimage.jpg")

wandb.log({"example": wandb.Image("myimage.jpg")})
```
  </TabItem>
</Tabs>

## 이미지 오버레이

<Tabs
  defaultValue="segmentation_masks"
  values={[
    {label: '분할 마스크', value: 'segmentation_masks'},
    {label: '경계 상자', value: 'bounding_boxes'},
  ]}>
  <TabItem value="segmentation_masks">

semantic segmentation 마스크를 로깅하고 W&B UI를 통해 상호 작용합니다(불투명도 변경, 시간에 따른 변화 보기 등).

![W&B UI에서의 인터랙티브 마스크 보기.](/images/track/semantic_segmentation.gif)

오버레이를 로깅하려면, 다음 키와 값으로 구성된 사전을 `wandb.Image`의 `masks` 키워드 인수에 제공해야 합니다:

* 이미지 마스크를 나타내는 두 가지 키 중 하나:
  * `"mask_data"`: 각 픽셀에 대한 정수 클래스 레이블이 포함된 2D NumPy 배열
  * `"path"`: (문자열) 저장된 이미지 마스크 파일의 경로
* `"class_labels"`: (선택적) 이미지 마스크에 있는 정수 클래스 레이블을 읽을 수 있는 클래스 이름에 매핑하는 사전

여러 마스크를 로깅하려면, 아래 코드 조각과 같이 여러 키가 있는 마스크 사전을 로깅하세요.

[실시간 예시 보기 →](https://app.wandb.ai/stacey/deep-drive/reports/Image-Masks-for-Semantic-Segmentation--Vmlldzo4MTUwMw)

[샘플 코드 →](https://colab.research.google.com/drive/1SOVl3EvW82Q4QKJXX6JtHye4wFix\_P4J)

```python
mask_data = np.array([[1, 2, 2, ..., 2, 2, 1], ...])

class_labels = {1: "tree", 2: "car", 3: "road"}

mask_img = wandb.Image(
    image,
    masks={
        "predictions": {"mask_data": mask_data, "class_labels": class_labels},
        "ground_truth": {
            # ...
        },
        # ...
    },
)
```
  </TabItem>
  <TabItem value="bounding_boxes">
이미지와 함께 경계 상자를 로깅하고, 필터와 토글을 사용하여 UI에서 다양한 상자 세트를 동적으로 시각화하세요.

![](@site/static/images/track/bb-docs.jpeg)

[실시간 예시 보기 →](https://app.wandb.ai/stacey/yolo-drive/reports/Bounding-Boxes-for-Object-Detection--Vmlldzo4Nzg4MQ)

경계 상자를 로깅하려면, 다음 키와 값으로 구성된 사전을 `wandb.Image`의 `boxes` 키워드 인수에 제공해야 합니다:

* `box_data`: 각 상자에 대한 사전이 하나씩 있는 사전 목록. 상자 사전 형식은 아래에 설명되어 있습니다.
  * `position`: 아래에 설명된 두 가지 형식 중 하나로 상자의 위치와 크기를 나타내는 사전. 모든 상자가 동일한 형식을 사용할 필요는 없습니다.
    * _옵션 1:_ `{"minX", "maxX", "minY", "maxY"}`. 각 상자 차원의 상한선과 하한선을 정의하는 좌표 세트를 제공합니다.
    * _옵션 2:_ `{"middle", "width", "height"}`. `middle` 좌표를 `[x,y]`로, `width`와 `height`를 스칼라로 지정하는 좌표 세트를 제공합니다.
  * `class_id`: 상자의 클래스 신원을 나타내는 정수. 아래의 `class_labels` 키를 참조하세요.
  * `scores`: UI에서 상자를 필터링하는 데 사용될 수 있는 문자열 레이블과 숫자 값으로 구성된 점수 사전.
  * `domain`: 상자 좌표의 단위/형식을 지정합니다. 상자 좌표가 픽셀 공간(즉, 이미지 치수의 한계 내에서 정수)으로 표현되는 경우 **"픽셀"로 설정하세요**. 기본적으로 도메인은 이미지의 분수/퍼센트(0과 1 사이의 부동 소수점 숫자)로 가정됩니다.
  * `box_caption`: (선택적) 이 상자에 표시될 레이블 텍스트로 사용될 문자열
* `class_labels`: (선택적) `class_id`를 문자열에 매핑하는 사전. 기본적으로 우리는 `class_0`, `class_1` 등의 클래스 레이블을 생성합니다.

이 예제를 확인하세요:

```python
class_id_to_label = {
    1: "car",
    2: "road",
    3: "building",
    # ...
}

img = wandb.Image(
    image,
    boxes={
        "predictions": {
            "box_data": [
                {
                    # 기본 상대적/분수 도메인에서 표현된 하나의 상자
                    "position": {"minX": 0.1, "maxX": 0.2, "minY": 0.3, "maxY": 0.4},
                    "class_id": 2,
                    "box_caption": class_id_to_label[2],
                    "scores": {"acc": 0.1, "loss": 1.2},
                    # 픽셀 도메인에서 표현된 또 다른 상자
                    # (설명 목적으로만, 모든 상자는 동일한 도메인/형식에 있을 가능성이 높음)
                    "position": {"middle": [150, 20], "width": 68, "height": 112},
                    "domain": "pixel",
                    "class_id": 3,
                    "box_caption": "a building",
                    "scores": {"acc": 0.5, "loss": 0.7},
                    # ...
                    # 필요한 만큼 많은 상자를 로깅하세요
                }
            ],
            "class_labels": class_id_to_label,
        },
        # 각 의미 있는 상자 그룹을 고유한 키 이름으로 로깅하세요
        "ground_truth": {
            # ...
        },
    },
)

wandb.log({"driving_scene": img})
```
  </TabItem>
</Tabs>

## 테이블에서의 이미지 오버레이

<Tabs
  defaultValue="segmentation_masks"
  values={[
    {label: '분할 마스크', value: 'segmentation_masks'},
    {label: '경계 상자', value: 'bounding_boxes'},
  ]}>
  <TabItem value="segmentation_masks">

![테이블에서의 인터랙티브 분할 마스크](/images/track/Segmentation_Masks.gif)

테이블에서 분할 마스크를 로깅하려면 테이블의 각 행에 대해 `wandb.Image` 객체를 제공해야 합니다.

아래에 코드 조각 예제가 제공되어 있습니다:

```python
table = wandb.Table(columns=["ID", "Image"])

for id, img, label in zip(ids, images, labels):
    mask_img = wandb.Image(
        img,
        masks={
            "prediction": {"mask_data": label, "class_labels": class_labels}
            # ...
        },
    )

    table.add_data(id, img)

wandb.log({"Table": table})
```
  </TabItem>
  <TabItem value="bounding_boxes">


![테이블에서의 인터랙티브 경계 상자](/images/track/Bounding_Boxes.gif)

테이블에서 이미지와 경계 상자를 로깅하려면 테이블의 각 행에 대해 `wandb.Image` 객체를 제공해야 합니다.

아래에 코드 조각 예제가 제공되어 있습니다:

```python
table = wandb.Table(columns=["ID", "Image"])

for id, img, boxes in zip(ids, images, boxes_set):
    box_img = wandb.Image(
        img,
        boxes={
            "prediction": {
                "box_data": [
                    {
                        "position": {
                            "minX": box["minX"],
                            "minY": box["minY"],
                            "maxX": box["maxX"],
                            "maxY": box["maxY"],
                        },
                        "class_id": box["class_id"],
                        "box_caption": box["caption"],
                        "domain": "pixel",
                    }
                    for box in boxes
                ],
                "class_labels": class_labels,
            }
        },
    )
```
  </TabItem>
</Tabs>

## 히스토그램

<Tabs
  defaultValue="histogram_logging"
  values={[
    {label: '기본 히스토그램 로깅', value: 'histogram_logging'},
    {label: '유연한 히스토그램 로깅', value: 'flexible_histogram'},
    {label: '요약에서의 히스토그램', value: 'histogram_summary'},
  ]}>
  <TabItem value="histogram_logging">
  
첫 번째 인수로 숫자 시퀀스(예: 리스트, 배열, 텐서)가 제공되면, `np.histogram`을 호출하여 자동으로 히스토그램을 구성합니다. 모든 배열/텐서는 펴집니다. 기본값인 `64`개의 구간을 재정의하려면 선택적 `num_bins` 키워드 인수를 사용할 수 있습니다. 지원되는 최대 구간 수는 `512`입니다.

UI에서는 학습 단계를 x축, 메트릭 값을 y축, 색상으로 표현된 수치로 히스토그램을 그려 학습 중에 로깅된 히스토그램을 비교하기 쉽게 합니다. "요약에서의 히스토그램" 탭에서 일회성 히스토그램 로깅에 대한 세부 사항을 확인하세요.

```python
wandb.log({"gradients": wandb.Histogram(grads)})
```

![GAN의 구분자에 대한 그레이디언트.](/images/track/histograms.png)
  </TabItem>
  <TabItem value="flexible_histogram">

더 많은 제어를 원하면 `np.histogram`을 호출하고 반환된 튜플을 `np_histogram` 키워드 인수로 전달하세요.

```python
np_hist_grads = np.histogram(grads, density=True, range=(0.0, 1.0))
wandb.log({"gradients": wandb.Histogram(np_hist_grads)})
```
  </TabItem>
  <TabItem value="histogram_summary">

```python
wandb.run.summary.update(  # 요약에만 있을 경우, Overview 탭에서만 보임
    {"final_logits": wandb.Histogram(logits)}
)
```
  </TabItem>
</Tabs>

히스토그램이 요약에 있으면 [실행 페이지](../../app/pages/run-page.md)의 Overview 탭에 표시됩니다. 이력에 있으면 차트 탭에서 시간이 지남에 따른 구간의 히트맵을 그립니다.

## 3D 시각화

<Tabs
  defaultValue="3d_object"
  values={[
    {label: '3D 개체', value: '3d_object'},
    {label: '포인트 클라우드', value: 'point_clouds'},
    {label: '분자', value: 'molecules'},
  ]}>
  <TabItem value="3d_object">

`'obj', 'gltf', 'glb', 'babylon', 'stl', 'pts.json'` 형식의 로그 파일을 업로드하면 실행이 완료될 때 UI에서 렌더링됩니다.

```python
wandb.log(
    {
        "generated_samples": [
            wandb.Object3D(open("sample.obj")),
            wandb.Object3D(open("sample.gltf")),
            wandb.Object3D(open("sample.glb")),
        ]
    }
)
```

![헤드폰 포인트 클라우드의 실제값 및 예측값](/images/track/ground_truth_prediction_of_3d_point_clouds.png)

[실시간 예제 보기 →](https://app.wandb.ai/nbaryd/SparseConvNet-examples\_3d\_segmentation/reports/Point-Clouds--Vmlldzo4ODcyMA)
  </TabItem>
  <TabItem value="point_clouds">

3D 포인트 클라우드와 리다(Lidar) 장면을 바운딩 박스와 함께 로그합니다. 점들의 좌표 및 색상을 포함하는 NumPy 배열을 전달하여 렌더링합니다. UI에서는 300,000점으로 제한합니다.

```python
point_cloud = np.array([[0, 0, 0, COLOR]])

wandb.log({"point_cloud": wandb.Object3D(point_cloud)})
```

색상 스키마를 유연하게 지원하기 위해 세 가지 형태의 NumPy 배열을 지원합니다.

* `[[x, y, z], ...]` `nx3`
* `[[x, y, z, c], ...]` `nx4` `| c는 범위 `[1, 14]` 내의 카테고리입니다` (세분화에 유용)
* `[[x, y, z, r, g, b], ...]` `nx6 | r,g,b`는 빨간색, 초록색, 파란색 채널의 값으로 `[0,255]` 범위입니다.

아래는 로깅 코드의 예시입니다:

* `points`는 위에 표시된 간단한 포인트 클라우드 렌더러와 같은 형식의 NumPy 배열입니다.
* `boxes`는 세 가지 속성을 가진 파이썬 사전의 NumPy 배열입니다:
  * `corners`- 여덟 모서리의 목록
  * `label`- 박스에 렌더링할 라벨을 나타내는 문자열 (선택 사항)
  * `color`- 박스의 색상을 나타내는 rgb 값
* `type`은 렌더링할 장면 유형을 나타내는 문자열입니다. 현재 지원되는 값은 `lidar/beta`뿐입니다

```python
# W&B에서 점과 박스 로깅
point_scene = wandb.Object3D(
    {
        "type": "lidar/beta",
        "points": np.array(  # 포인트 클라우드처럼 점 추가
            [[0.4, 1, 1.3], [1, 1, 1], [1.2, 1, 1.2]]
        ),
        "boxes": np.array(  # 3d 박스 그리기
            [
                {
                    "corners": [
                        [0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [1, 0, 0],
                        [1, 1, 0],
                        [0, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1],
                    ],
                    "label": "Box",
                    "color": [123, 321, 111],
                },
                {
                    "corners": [
                        [0, 0, 0],
                        [0, 2, 0],
                        [0, 0, 2],
                        [2, 0, 0],
                        [2, 2, 0],
                        [0, 2, 2],
                        [2, 0, 2],
                        [2, 2, 2],
                    ],
                    "label": "Box-2",
                    "color": [111, 321, 0],
                },
            ]
        ),
        "vectors": np.array(  # 3D 벡터 추가
            [{"start": [0, 0, 0], "end": [0.1, 0.2, 0.5]}]
        ),
    }
)
wandb.log({"point_scene": point_scene})
```
  </TabItem>
  <TabItem value="molecules">

```python
wandb.log({"protein": wandb.Molecule("6lu7.pdb")})
```

`pdb`, `pqr`, `mmcif`, `mcif`, `cif`, `sdf`, `sd`, `gro`, `mol2`, `mmtf` 등 10가지 파일 유형의 분자 데이터를 로그합니다.

W&B는 SMILES 문자열, [`rdkit`](https://www.rdkit.org/docs/index.html) `mol` 파일 및 `rdkit.Chem.rdchem.Mol` 개체에서 분자 데이터를 로깅하는 것도 지원합니다.

```python
resveratrol = rdkit.Chem.MolFromSmiles("Oc1ccc(cc1)C=Cc1cc(O)cc(c1)O")

wandb.log(
    {
        "resveratrol": wandb.Molecule.from_rdkit(resveratrol),
        "녹색 형광 단백질": wandb.Molecule.from_rdkit("2b3p.mol"),
        "아세트아미노펜": wandb.Molecule.from_smiles("CC(=O)Nc1ccc(O)cc1"),
    }
)
```

실행이 완료되면 UI에서 분자의 3D 시각화와 상호작용할 수 있습니다.

[AlphaFold를 사용한 실시간 예제 보기 →](http://wandb.me/alphafold-workspace)

![](@site/static/images/track/docs-molecule.png)
  </TabItem>
</Tabs>

## 기타 미디어

W&B는 다양한 다른 미디어 유형의 로깅도 지원합니다.

<Tabs
  defaultValue="audio"
  values={[
    {label: '오디오', value: 'audio'},
    {label: '비디오', value: 'video'},
    {label: '텍스트', value: 'text'},
    {label: 'HTML', value: 'html'},
  ]}>
  <TabItem value="audio">

```python
wandb.log({"고래 노래": wandb.Audio(np_array, caption="OooOoo", sample_rate=32)})
```

단계별로 로그할 수 있는 오디오 클립의 최대 수는 100개입니다.

  </TabItem>
  <TabItem value="video">

```python
wandb.log({"video": wandb.Video(numpy_array_or_path_to_video, fps=4, format="gif")})
```

Numpy 배열을 제공하는 경우 차례대로 시간, 채널, 너비, 높이의 차원임을 가정합니다. 기본적으로 4fps gif 이미지를 생성합니다([`ffmpeg`](https://www.ffmpeg.org) 및 [`moviepy`](https://pypi.org/project/moviepy/) 파이썬 라이브러리가 numpy 객체를 전달할 때 필요). 지원되는 형식은 `"gif"`, `"mp4"`, `"webm"`, `"ogg"`입니다. `wandb.Video`에 문자열을 전달하면 파일이 존재하고 지원되는 형식인지 확인한 후 wandb에 업로드합니다. `BytesIO` 개체를 전달하면 지정된 형식의 확장명으로 임시 파일을 생성합니다.

W&B [실행](../../app/pages/run-page.md) 및 [프로젝트](../../app/pages/project-page.md) 페이지에서 미디어 섹션에서 비디오를 볼 수 있습니다.

  </TabItem>
  <TabItem value="text">

UI에서 테이블 형식으로 텍스트를 로그하려면 `wandb.Table`을 사용하세요. 기본적으로 열 헤더는 `["Input", "Output", "Expected"]`입니다. 최적의 UI 성능을 보장하기 위해 기본적으로 최대 행 수는 10,000으로 설정되어 있습니다. 그러나 사용자는 `wandb.Table.MAX_ROWS = {원하는 최대값}`으로 최대값을 명시적으로 재정의할 수 있습니다.

```python
columns = ["Text", "Predicted Sentiment", "True Sentiment"]
# 방법 1
data = [["I love my phone", "1", "1"], ["My phone sucks", "0", "-1"]]
table = wandb.Table(data=data, columns=columns)
wandb.log({"examples": table})

# 방법 2
table = wandb.Table(columns=columns)
table.add_data("I love my phone", "1", "1")
table.add_data("My phone sucks", "0", "-1")
wandb.log({"examples": table})
```

pandas `DataFrame` 개체도 전달할 수 있습니다.

```python
table = wandb.Table(dataframe=my_dataframe)
```
  </TabItem>
  <TabItem value="html">

```python
wandb.log({"custom_file": wandb.Html(open("some.html"))})
wandb.log({"custom_string": wandb.Html('<a href="https://mysite">Link</a>')})
```

사용자 지정 html은 모든 키에서 로그할 수 있으며, 실행 페이지에 HTML 패널을 노출합니다. 기본적으로 기본 스타일을 주입하지만, `inject=False`를 전달하여 기본 스타일을 비활성화할 수 있습니다.

```python
wandb.log({"custom_file": wandb.Html(open("some.html"), inject=False)})
```

  </TabItem>
</Tabs>

## 자주 묻는 질문

### 에포크나 단계별로 이미지나 미디어를 어떻게 비교하나요?

단계별로 이미지를 로그할 때마다 UI에 저장하여 표시합니다. 이미지 패널을 확장하고 단계 슬라이더를 사용하여 다른 단계의 이미지를 살펴보세요. 이를 통해 모델의 출력이 학습하는 동안 어떻게 변하는지 쉽게 비교할 수 있습니다.

### 이미지나 미디어를 업로드하지 않고 W&B를 프로젝트에 통합하고 싶다면 어떻게 해야 하나요?

스칼라만 로그하는 프로젝트의 경우에도 W&B를 사용할 수 있습니다 — 업로드하고 싶은 파일이나 데이터를 명시적으로 지정합니다. 여기 [PyTorch에서의 간단한 예제](http://wandb.me/pytorch-colab)가 있으며, 이미지를 로그하지 않습니다.

### PNG를 어떻게 로그하나요?

[`wandb.Image`](../../../ref/python/data-types/image.md)는 기본적으로 `numpy` 배열이나 `PILImage` 인스턴스를 PNG로 변환합니다.

```python
wandb.log({"example": wandb.Image(...)})
# 또는 여러 이미지
wandb.log({"example": [wandb.Image(...) for img in images]})
```

### 비디오를 어떻게 로그하나요?

비디오는 [`wandb.Video`](../../../ref/python/data-types/video.md) 데이터 유형을 사용하여 로그됩니다:

```python
wandb.log({"example": wandb.Video("myvideo.mp4")})
```

이제 프로젝트 워크스페이스, 실행 워크스페이스 또는 리포트에서 "시각화 추가"를 클릭하여 리치 미디어 패널에 비디오를 볼 수 있습니다.

### 포인트 클라우드를 내비게이션하고 확대하는 방법은 무엇인가요?

컨트롤을 누르고 마우스를 사용하여 공간 내에서 이동할 수 있습니다.

### 분자의 2D 뷰를 어떻게 로그하나요?

[`wandb.Image`](../../../ref/python/data-types/image.md) 데이터 유형과 [`rdkit`](https://www.rdkit.org/docs/index.html)을 사용하여 분자의 2D 뷰를 로그할 수 있습니다:

```python
molecule = rdkit.Chem.MolFromSmiles("CC(=O)O")
rdkit.Chem.AllChem.Compute2DCoords(molecule)
rdkit.Chem.AllChem.GenerateDepictionMatching2DStructure(molecule, molecule)
pil_image = rdkit.Chem.Draw.MolToImage(molecule, size=(300, 300))

wandb.log({"아세트산": wandb.Image(pil_image)})
```
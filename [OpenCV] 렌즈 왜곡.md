# 5.3 렌즈 왜곡

행렬식으로 표현할 수 없는 모양의 변환도 필요할 때가 있다. 투명한 물잔을 통해 비친 장면이나 일렁이는 물결에 반사된 모습 같은 것이 대표적이다. 이런 렌즈 왜곡 변환에 대해서 알아보자.

### 5.3.1 리매핑

OpenCV는 규칙성 없이 마음대로 모양을 변환해 주는 함수로 **cv2.remap()**을 제공한다. 이 함수는 기존 픽셀의 위치를 원하는 위치로 재배치한다.

- dst = cv2.remap(src, mapx, mapy, interpolation [, dst, borderMode, borderValue])
    - src : 입력 영상
    - mapx, mapy : x축과 y축으로 이동할 좌표(인덱스), src와 동일한 크기, dtype=float32
    - 나머지 인자는 cv2.warpAffine()과 동일
    - dst : 결과 영상

mapx, mapy는 초기값으로 0이나 1같은 의미 없는 값이 아니라 영상의 원래 좌표값을 가지고 있는 것이 좋다. 전체 픽셀 중에 옮기고 싶은 몇몇 픽셀에 대해서만 새로운 좌표를 지정하거나 원래 있던 위치에서 얼마만큼 움직이라고 하는 것이 코딩하기 편하기 때문이다. 

**np.indices()** 함수를 사용하기 → 배열을 주어진 크기로 생성, 자신의 인덱스를 값으로 초기화해서 3차원 배열로 반환한다.

```python
mapy, mapx = np.indices((rows, cols), dtype=np.float32)
```

cv2.remap() 함수는 재배치할 좌표가 정수로 떨어지지 않는 등의 이유로 픽셀이 탈락하는 경우 자동으로 보정을 수행한다.

다음 예제에서는 영상을 뒤집는 작업을 cv2.warpAffine() 함수와 cv2.remap() 함수로 각각 구현해본다.

> **[예제 5-10] 변환행렬과 리매핑으로 영상 뒤집기(5.10_remap_flip.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> import time
> import matplotlib.pyplot as plt
> 
> img = cv2.imread('./img/girl.jpg')
> rows, cols = img.shape[:2]
> 
> # 뒤집기 변환행렬로 구현
> st = time.time()
> mflip = np.float32([[-1, 0, cols-1], [0, -1, rows-1]])  # 변환행렬 생성
> fliped1 = cv2.warpAffine(img, mflip, (cols, rows))      # 변환 적용
> print('matrix:', time.time()-st)
> 
> >>> matrix: 0.0017158985137939453
> 
> # remap 함수로 뒤집기 구현
> st2 = time.time()
> mapy, mapx = np.indices((rows, cols), dtype=np.float32)  # 매핑 배열 초기화 생성
> mapx = cols - mapx - 1                                   # x축 좌표 뒤집기 연산
> mapy = rows - mapy - 1                                   # y축 좌표 뒤집기 연산
> fliped2 = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)   # remap 적용
> print('remap:', time.time()-st2)
> 
> >>> remap: 0.010727167129516602
> 
> # 결과 출력
> imgs = {'origin':img, 'fliped1':fliped1, 'fliped2':fliped2}
> 
> for i, (k, v) in enumerate(imgs.items()):
>     plt.subplot(1, 3, i+1)
>     plt.title(k)
>     plt.imshow(v[:,:,::-1])
>     plt.xticks([])
>     plt.yticks([])
> 
> plt.show()
> ```
> 
> ![Untitled](https://user-images.githubusercontent.com/69300448/223967685-bd553a0d-86b1-45a2-bddb-c11b847e34aa.png)
> 

행렬식으로 표현할 수 있는 변환을 cv2.remap() 함수로 변환하는 것은 코드도 복잡하고 수행속도도 드린 것을 볼 수 있다. 따라서 cv2.remap() 함수는 변환행렬로 표현할 수 없는 비선형 변환에만 사용하는 것이 좋다. 영상 분야에서 비선형 변환은 대부분 렌즈 왜곡과 관련한 보정이나 효과에 사용한다.

> **[예제 5-11] 삼각함수를 이용한 비선형 리매핑(5.11_remap_sin_cos.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> import matplotlib.pyplot as plt
> 
> l = 20       # 파장(wave length)
> amp = 15     # 진폭(amplitude)
> 
> img = cv2.imread('./img/taekwonv1.jpg')
> rows, cols = img.shape[:2]
> 
> # 초기 매핑 배열 생성
> mapy, mapx = np.indices((rows, cols), dtype=np.float32)
> 
> # sin, cos 함수를 이용한 매핑 연산
> sinx = mapx + amp * np.sin(mapy/l)
> cosy = mapy + amp * np.cos(mapx/l)
> 
> # 영상 리매핑
> img_sinx = cv2.remap(img, sinx, mapy, cv2.INTER_LINEAR)   # x축만 sin 곡선 적용
> img_cosy = cv2.remap(img, mapx, cosy, cv2.INTER_LINEAR)   # y축만 cos 곡선 적용
> 
> # x, y 축 모두 sin, cos 곡선 적용 및 외곽 영역 보정
> img_both = cv2.remap(img, sinx, cosy, cv2.INTER_LINEAR, None, cv2.BORDER_REPLICATE)
> 
> # 결과 출력
> imgs = {'origin':img, 'sin x':img_sinx, 'cos y':img_cosy, 'sin cos':img_both}
> 
> fig = plt.figure(figsize=(10, 5))
> for i, (k, v) in enumerate(imgs.items()):
>     plt.subplot(1, 4, i+1)
>     plt.title(k)
>     plt.imshow(v[:,:,::-1])
>     plt.xticks([])
>     plt.yticks([])
> plt.show()
> ```
> 
> ![Untitled 1](https://user-images.githubusercontent.com/69300448/223967733-ff071c91-04ba-4580-8c4c-2b35c8106f20.png)
> 

### 5.3.2 오목 렌즈와 볼록 렌즈 왜곡

리매핑 함수를 이용해서 볼록 렌즈와 오목 렌즈 효과를 만들어보자. 

원을 대상으로 작업을 하기 전에 좌표계에 대한 이해를 먼저 해보자.

원 안의 점 $p$를 가리킬 때 두 가지 방법으로 표현할 수 있다.

- 좌표 $(x, y)$로 나타내는 **직교좌표계(cartesian coordinate system)**
- 원점 $O$에서 점 $p$까지의 거리와 각 $\theta$를 이용해 $(r, \theta)$로 표현하는 **극좌표계(polar coordinate system)**

두 좌표계는 다음 식으로 상호 변환을 할 수있다.

- 직교좌표 → 극좌표 : $\theta = arctan(x, y), r = \sqrt{x^2 + y^2}$
- 극좌표 → 직교좌표 : $x = rcos(\theta), y = rsin(\theta)$

OpenCV는 좌표 변환을 위한 함수를 제공한다.

- r, theta = cv2.cartToPolar(x, y) : 직교좌표 → 극좌표 변환
- x, y = cv2.polarToCart(r, theta) : 극좌표 → 직교좌표 변환
    - x, y : x, y 좌표 배열
    - r : 원점과의 거리
    - theta : 각도 값

직교좌표를 극좌표로 바꾸면 원 안의 픽셀만을 대상으로 손쉽게 작업할 수 있고, 원점과의 거리 $r$에 연산을 해서 원의 모양이나 범위를 계산하기 쉽다.

좌표의 변환뿐만 아니라 좌표의 기준점도 변경하는 것이 연산에 유리하다. 영상의 경우 직교좌표를 사용할 때는 좌상단 끝을 (0, 0) 좌표로 하는데, 극좌표를 사용하는 경우에는 영상의 중앙을 기준점으로 사용하는 것이 당연하고, 원점을 기준으로 좌측과 하단은 음수 좌표가 필요하므로 좌표의 값도 -1 ~ 1로 노멀라이즈해서 사용하는 것이 편리하다.

> **[예제 5-12] 볼록/오목 렌즈 왜곡 효과(5.12_remap_lens.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> img = cv2.imread('./img/taekwonv1.jpg')
> rows, cols = img.shape[:2]
> 
> # 설정값 세팅
> exp = 0.5    # 볼록, 오목 지수(오목 : 0.1 ~ 1, 볼록 : 1.1 ~)
> exp2 = 1.5
> scale = 1    # 변환 영역 크기(0 ~ 1)
> 
> # 매핑 배열 생성
> mapy, mapx = np.indices((rows, cols), dtype=np.float32)
> 
> # 좌상단 기준좌표에서 -1 ~ 1로 정규화된 중심점 기준 좌표로 변경
> mapx = 2*mapx/(cols - 1) - 1
> mapy = 2*mapy/(rows - 1) - 1
> 
> # 직교좌표를 극좌표로 변환
> r, theta = cv2.cartToPolar(mapx, mapy)
> r2, theta2 = cv2.cartToPolar(mapx, mapy)
> 
> # 왜곡 영역만 중심 확대/축소 지수 적용
> r[r<scale] = r[r<scale] ** exp
> r2[r2<scale] = r2[r2<scale] ** exp2
> 
> # 극좌표를 직교좌표로 변환
> mapx, mapy = cv2.polarToCart(r, theta)
> mapx2, mapy2 = cv2.polarToCart(r2, theta)
> 
> # 중심점 기준에서 좌상단 기준으로 변경
> mapx = ((mapx + 1) * cols - 1)/2
> mapy = ((mapy + 1) * rows - 1)/2
> 
> mapx2 = ((mapx2 + 1) * cols - 1)/2
> mapy2 = ((mapy2 + 1) * rows - 1)/2
> 
> # 리매핑 변환
> distorted = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
> distorted2 = cv2.remap(img, mapx2, mapy2, cv2.INTER_LINEAR)
> 
> # 결과 출력
> cv2.imshow('origin', img)
> cv2.imshow('distorted', distorted)
> cv2.imshow('distorted2', distorted2)
> cv2.waitKey()
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled 2](https://user-images.githubusercontent.com/69300448/223967787-48c8dbd0-13b4-4f6d-b6c1-42fdedae089b.png)
> 

exp로 지수 연산을 하면 지수 값이 1보다 큰 경우 중심점의 밀도는 낮아지고 외곽으로 갈수록 밀도가 높아지는 반면, 지수 값이 1보다 작은 경우에는 그 반대가 된다. 밀도가 낮아지면 픽셀 간의 거리가 멀어진 것이므로 확대한 효과와 같아지고, 밀도가 높아지면 픽셀 간의 거리가 원래보다 가까워진 것이므로 축소한 효과와 같다.

### 5.3.3 방사 왜곡

카메라 렌즈는 동그랗고 영상은 사각형이라서 렌즈 가장자리 영상에는 왜곡이 생기게 된다. 이런 왜곡을 **배럴 왜곡(barrel distortion)**이라고 한다. 배럴 왜곡을 해결하기 위해서 다음과 같은 수학식이 나오게 되었다.

$$
r_d = r_u ( 1 + k_1r^2_u + k_2r_u^4 + k_3r_u^6)
$$

- $r_d$ : 왜곡 변형 후
- $r_u$ : 왜곡 변형 전
- $k_1, k_2, k_3$ : 왜곡 계수

배럴 왜곡의 왜곡 계수의 값에 따라 밖으로 튀어나오는 배럴 왜곡이 나타나기도 하고, 안으로 들어가는 핀쿠션 왜곡(pincushion distortion)이 일어나기도 한다.

> **[예제 5-13] 방사 왜곡 효과(5.13_remap_barrel.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> # 왜곡 계수 설정
> k1, k2, k3 = 0.5, 0.2, 0.0   # 배럴 왜곡
> # k1, k2, k3 = -0.3, 0, 0      # 핀쿠션 왜곡
> 
> img = cv2.imread('./img/girl.jpg')
> rows, cols = img.shape[:2]
> 
> # 매핑 배열 생성
> mapy, mapx = np.indices((rows, cols), dtype=np.float32)
> 
> # 중앙점 좌표로 -1~1 정규화 및 극좌표 변환
> mapx = 2*mapx/(cols - 1) - 1
> mapy = 2*mapy/(rows - 1) - 1
> r, theta = cv2.cartToPolar(mapx, mapy)
> 
> # 방사 왜곡 변형 연산
> ru = r*(1+k1*(r**2) + k2*(r**4) + k3*(r**6))
> 
> # 직교좌표 및 좌상단 기준으로 복원
> mapx, mapy = cv2.polarToCart(ru, theta)
> mapx = ((mapx + 1)*cols -1)/2
> mapy = ((mapy + 1)*rows -1)/2
> 
> # 리매핑
> distorted = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
> 
> # 결과 출력
> cv2.imshow('original', img)
> cv2.imshow('distorted', distorted)
> cv2.waitKey()
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled 3](https://user-images.githubusercontent.com/69300448/223967840-308746d3-f73d-4b98-9edf-9fd0a33fbb90.png)
> 

OpenCV는 배럴 왜곡 현상이 일어나는 렌즈의 왜곡을 제거할 목적으로 **cv2.undistort()** 함수를 제공한다.

- **dst = cv2.undistort(src, cameraMtrix, distCoeffs)**
    - src : 입력 원본 영상
    - cameraMatrix : 카메라 메트릭스
        
        $\begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$
        
    - distCoeffs : 왜곡 계수, 최소 4개 또는 5, 8, 12, 14개
        - (k1, k2, p1, p2[, k3])

> **[예제 5-14] 방사 왜곡 효과(5.14_undistort_barrel.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> import matplotlib.pyplot as plt
> 
> # 격자 무늬 이미지 생성
> img = np.full((300, 400, 3), 255, np.uint8)
> img[::10, :, :] = 0
> img[:, ::10, :] = 0
> width = img.shape[1]
> height = img.shape[0]
> 
> # 왜곡 계수 설정
> k1, k2, p1, p2 = 0.001, 0, 0, 0    # 배럴 왜곡
> # k1, k2, p1, p2 = -0.0005, 0, 0, 0  # 핀쿠션 왜곡
> distCoeff = np.float64([k1, k2, p1, p2])
> 
> # 임의의 값으로 카메라 매트릭스 설정
> fx, fy = 10, 10
> cx, cy = width/2, height/2
> camMtx = np.float32([[fx, 0, cx],
>                     [0, fy, cy],
>                     [0, 0, 1]])
> 
> # 왜곡 변형
> dst = cv2.undistort(img, camMtx, distCoeff)
> 
> cv2.imshow('original', img)
> cv2.imshow('dst', dst)
> cv2.waitKey()
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled 4](https://user-images.githubusercontent.com/69300448/223967889-bcb6e51e-a71a-4e18-8c73-c53d1143b9eb.png)
>

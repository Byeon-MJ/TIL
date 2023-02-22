# 4.3 스레시홀딩

이미지를 검은색과 흰색만으로 표현한 것을 **바이너리 이미지**라고 한다. 이미지에서 원하는 피사체의 모양을 좀 더 정확히 판단하기 위해서 사용한다. (예 : 종이에서 글씨만을 분리, 배경과 전경을 분리)

**스레시홀딩(Thresholding)** : 값을 경계점을 기준으로 두 가지 부류로 나누는 것, 바이너리 이미지를 만드는 가장 대표적인 방법

### 4.3.1 전역 스레시 홀딩

바이너리 이미지를 만들기 위해서 컬러 이미지를 그레이 스케일로 바꾸고 각 픽셀의 값이 경계값을 넘으면 255, 넘지 못하면 0을 지정한다.

간단한 Numpy 연산만으로도 충분히 가능하지만 OpenCV는 **cv2.threshold() 함수**로 더 많은 기능을 제공한다.

> **[예제 4-9] 바이너리 이미지 만들기(4.9_threshold.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> import matplotlib.pyplot as plt
> 
> # 이미지 그레이 스케일로 읽기
> img = cv2.imread('./img/gray_gradient.jpg', cv2.IMREAD_GRAYSCALE)
> 
> # Numpy 연산으로 바이너리 이미지 만들기
> thresh_np = np.zeros_like(img)  # 원본과 동일한 0으로 채워진 이미지
> thresh_np[img > 127] = 255      # 127보다 큰 값만 255로 변경
> 
> # OpenCV 함수로 바이너리 이미지 만들기
> ret, thresh_cv = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
> print(ret)
> 
> >>> 127.0
> 
> # 원본과 결과물 출력
> imgs ={'Original':img, 'Numpy API':thresh_np, 'cv2.threshold':thresh_cv}
> 
> for i, (key, value) in enumerate(imgs.items()):
>     plt.subplot(1, 3, i+1)
>     plt.title(key)
>     plt.imshow(value, cmap='gray')
>     plt.xticks([])
>     plt.yticks([])
> 
> plt.show()
> ```
> 
> ![Untitled](4%203%20%E1%84%89%E1%85%B3%E1%84%85%E1%85%A6%E1%84%89%E1%85%B5%E1%84%92%E1%85%A9%E1%86%AF%E1%84%83%E1%85%B5%E1%86%BC%20a3ad5fa1f3414164a95cea6bd2319fb6/Untitled.png)
> 
- **ret, out = cv2.threshold(img, threshold, value, type_flag)**
    - img : Numpy 배열, 변환할 이미지
    - threshold : 경계값
    - value : 경계값 기준에 만족하는 픽셀에 적용할 값
    - type_flag : 스레시홀드 적용 방법 지정
        - cv2.THRESH_BINARY : px > threshold ? value : 0, 픽셀 값이 경계값을 넘으면 value를 지정하고, 넘지 못하면 0을 지정
        - cv2.THRESH_BIANARY_INV : px > threshold ? 0 : value, cv2.THRESH_BINARY의 반대
        - cv2.THRESH_TRUNC : px > threshold ? threshold : px, 픽셀 값이 경계값을 넘으면 경계값을 지정하고, 넘지 못하면 원래의 값 유지
        - cv2.THRESH_TOZERO : px > threshold ? px : 0, 픽셀 값이 경계값을 넘으면 원래 값을 유지, 넘지 못하면 0을 지정
        - cv2.THRESH_TOZERO_INV : px > threshold ? 0 : px, cv2.THRESH_TOZERO의 반대
    - ret : 스레시홀딩에 사용한 경계값
    - out : 결과 바이너리 이미지

> **[예제 4-10] 스레시홀딩 플래그 실습(4.10_threshold_flag.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> import matplotlib.pyplot as plt
> 
> img = cv2.imread('./img/gray_gradient.jpg', cv2.IMREAD_GRAYSCALE)
> 
> _, t_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
> _, t_bininv = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
> _, t_truc = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
> _, t_2zr = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
> _, t_2zrinv = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
> 
> imgs = {'Origin':img, 'BINARY':t_bin, 'BINARY_INV':t_bininv,
>        'TRUNC':t_truc, 'TOZERO':t_2zr, 'TOZERO_INV':t_2zrinv}
> 
> for i, (key, value) in enumerate(imgs.items()):
>     plt.subplot(2, 3, i+1)
>     plt.title(key)
>     plt.imshow(value, cmap='gray')
>     plt.xticks([])
>     plt.yticks([])
>     
> plt.show()
> ```
> 
> ![Untitled](4%203%20%E1%84%89%E1%85%B3%E1%84%85%E1%85%A6%E1%84%89%E1%85%B5%E1%84%92%E1%85%A9%E1%86%AF%E1%84%83%E1%85%B5%E1%86%BC%20a3ad5fa1f3414164a95cea6bd2319fb6/Untitled%201.png)
> 

### 4.3.2 오츠의 알고리즘

바이너리 이미지를 만들 때 경계값을 얼마로 정하느냐가 가장 중요하다.

적절한 경계값을 정하기 위해서 여러 차례에 걸쳐 경계값을 조금씩 수정해 가면서 찾아야 한다.

**오츠 노부유키(Nobuyuki Otsu)**는 반복적인 시도 없이 한 번에 효율적으로 경계값을 찾을 수 있는 방법을 제안, 그 이름을 따서 **오츠의 이진화 알고리즘(Otsu’s binarization method)** 라고 한다.

경계값을 임의로 정해서 픽셀들을 두 부류로 나누고 두 부류의 명암 분포를 반복해서 구한 다음 두 부류의 명암 분포를 가장 균일하게 하는 경계값을 선택하는 방법

$$
\sigma^2_w(t) = w_1(t)\sigma^2_1(t) + w_2(t)\sigma^2_2(t)
$$

- $t$ : 0~255, 경계값
- $w_1, w_2$ : 각 부류의 비율 가중치
- $\sigma^2_1, \sigma^2_2$ : 각 부류의 분산

cv2.threshold() 함수의 마지막 인자에 cv2.THRESH_OTSU를 추가해서 전달하면 사용할 수 있다.

원래 경계값을 전달해야 하는 두번째 인자 threshold는 무시되므로 아무 숫자나 전달해도 되고, 결과값으로 오츠의 알고리즘에 의해 선택된 경계값을 반환값 ret로 받을 수 있다.

마지막 플래그는 스레시홀드 방식을 결정하는 플래그와 파이프( | ) 문자로 연결하여 전달한다.

```python
ret, t_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
```

> **[예제 4-11] 오츠의 알고리즘을 적용한 스레시홀드(4.11_threshold_otsu.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> import matplotlib.pyplot as plt
> 
> # 이미지를 그레이 스케일로 읽기
> img = cv2.imread('./img/scaned_paper.jpg', cv2.IMREAD_GRAYSCALE)
> 
> # 경계값을 130으로 지정
> _, t_130 = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)
> 
> # 경계값을 지정하지 않고 오츠의 알고리즘 선택
> t, t_otsu = cv2.threshold(img, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
> print('otsu threshold:', t)
> 
> >>> otsu threshold: 131.0
> 
> imgs = {'Original':img, 't:130':t_130, f'otsu:{t}':t_otsu}
> 
> for i, (key, value) in enumerate(imgs.items()):
>     plt.subplot(1, 3, i+1)
>     plt.title(key)
>     plt.imshow(value, cmap='gray')
>     plt.xticks([])
>     plt.yticks([])
>     
> plt.show()
> ```
> 
> ![Untitled](4%203%20%E1%84%89%E1%85%B3%E1%84%85%E1%85%A6%E1%84%89%E1%85%B5%E1%84%92%E1%85%A9%E1%86%AF%E1%84%83%E1%85%B5%E1%86%BC%20a3ad5fa1f3414164a95cea6bd2319fb6/Untitled%202.png)
> 

다만, 오츠의 알고리즘은 모든 경우의 수에 대해 경계값을 조사해야 하므로 속도가 빠르지 못하다는 단점이 있다. 또한 노이즈가 많은 영상에는 오츠의 알고리즘을 적용해도 좋은 결과를 얻지 못하는 경우가 많다. 나중에 배울 블러링 필터를 먼저 적용해야 한다.

### 4.3.3 적응형 스레시홀드

원본 영상(또는 이미지)에 조명이 일정하지 않거나 배경색이 여러 가지인 경우에는 아무리 경계값을 바꿔가며 시도해도 하나의 경계값을 이미지 전체에 적용해서는 좋은 결과를 얻지 못한다.

이때 이미지를 여러 영역으로 나눈 다음 그 주변 픽셀 값만 가지고 계산을 해서 경계값을 구하는 **적응형 스레시홀드(adaptive threshold)**를 적용한다.

- cv2.adaptiveThreshold(img, value, method, type_flag, block_size, C)
    - img : 입력 영상
    - value : 경계값을 만족하는 픽셀에 적용할 값
    - method : 경계값 결정 방법
        - cv2.ADPTIVE_THRESH_MEAN_C : 이웃 픽셀의 평균으로 결정
        - cv2.ADPTIVE_THRESH_GAUSSIAN_C : 가우시안 분포에 따른 가중치의 합으로 결정
    - type_flag : 스레시홀드 적용 방법 지정(cv2.threshold() 함수와 동일)
    - block_size : 영역으로 나눌 이웃의 크기(n x n), 홀수(3, 5, 7, …)
    - C : 계산된 경계값 결과에서 가감할 상수(음수 가능)

> **[예제 4-12] 적응형 스레시홀드 적용(4.12_thresh_adapted.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> import matplotlib.pyplot as plt
> 
> blk_size = 9
> C = 5
> img = cv2.imread('./img/sudoku.png', cv2.IMREAD_GRAYSCALE)
> 
> # 오츠의 알고리즘으로 단일 경계값을 전체 이미지에 적용
> ret, th1 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
> 
> # 적응형 스레시홀드를 평균으로 적용
> th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
>                            cv2.THRESH_BINARY, blk_size, C)
> 
> # 적응형 스레시홀드를 가우시안 분포로 적용
> th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
>                            cv2.THRESH_BINARY, blk_size, C)
> 
> # 결과를 출력
> imgs = {'Original':img, f'Global-Otsu:{ret}':th1, 
> 				'Adapted-Mean':th2, 'Adapted-Gaussian':th3}
> 
> for i, (k, v) in enumerate(imgs.items()):
>     plt.subplot(2, 2, i+1)
>     plt.title(k)
>     plt.imshow(v, 'gray')
>     plt.xticks([])
>     plt.yticks([])
>     
> plt.show()
> ```
> 
> ![Untitled](4%203%20%E1%84%89%E1%85%B3%E1%84%85%E1%85%A6%E1%84%89%E1%85%B5%E1%84%92%E1%85%A9%E1%86%AF%E1%84%83%E1%85%B5%E1%86%BC%20a3ad5fa1f3414164a95cea6bd2319fb6/Untitled%203.png)
> 

오츠의 알고리즘을 적용해서 96을 경계값으로 전체 이미지에 적용하면 좌측 하단은 검게 타버리고, 우측 상단은 하얗게 날아간다.

적응형 스레시홀드를 평균과 가우시안 분포를 각각 적용해서 더 좋은 겨로가를 얻을 수 있다.

가우시안 분포를 적용한 결과는 선명함은 떨어지지만 잡티(noise)가 훨씬 적은 것을 알 수 있다.

대부분의 이미지는 조명 차이와 그림자 때문에 지역적(local) 적용이 필요하다.
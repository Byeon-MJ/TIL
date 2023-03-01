# 4.5 히스토그램

**히스토그램(histogram)**은 무엇이 몇 개 있는지 그 수를 세어 놓은 것을 그림으로 표시한 것을 말한다. 히스토그램은 영상을 분석하는 데 도움이 많이 된다.

### 4.5.1 히스토그램 계산과 표시

영상 분야에서의 히스토그램은 전체 영상에서 픽셀 값이 1인 픽셀이 몇 개이고 2인 픽셀이 몇 개이고 하는 식으로 픽셀 값이 255인 픽셀이 몇 개인지까지 세는 것을 말한다. 전체 영상에서 픽셀들의 색상이나 명암의 분포를 파악할 수 있다.

OpenCV에서는 히스토그램을 계산하는 **cv2.calcHist()** 함수를 제공한다.

- **cv2.calcHist(img, channel, mask, histSize, ranges)**
    - img : 입력 영상, [img] 처럼 리스트로 감싸서 표현
    - channel : 처리할 채널, 리스트로 감싸서 표현
        - 1채널 : [0], 2채널 : [0, 1], 3채널 : [0, 1, 2]
    - mask : 마스크에 지정한 픽셀만 히스토그램 계산
    - histSize : 계급(bin)의 개수, 채널 개수에 맞게 리스트로 표현
        - 1채널 : [256], 2채널 : [256, 256], 3채널 : [256, 256, 256]
    - ranges : 각 픽셀이 가질 수 있는 값의 범위, RGB인 경우 [0, 256]
    

> **[예제 4-25] 그레이 스케일 1채널 히스토그램(4.25_histo_gray.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> import matplotlib.pyplot as plt
> 
> # 이미지 그레이 스케일로 읽기 및 출력
> img = cv2.imread('./img/mountain.jpg', cv2.IMREAD_GRAYSCALE)
> 
> plt.imshow(img, cmap='gray')
> plt.xticks([])
> plt.yticks([])
> plt.show()
> 
> # 히스토그램 계산 및 그리기
> hist = cv2.calcHist([img], [0], None, [256], [0, 256])
> plt.plot(hist)
> print('hist.shape :', hist.shape)
> print('hist.sum() :', hist.sum(), ', img.shape :', img.shape)
> plt.show()
> 
> >>> hist.shape : (256, 1)
> 		hist.sum() : 270000.0 , img.shape : (450, 600)
> ```
> 
> ![Untitled](4%205%20%E1%84%92%E1%85%B5%E1%84%89%E1%85%B3%E1%84%90%E1%85%A9%E1%84%80%E1%85%B3%E1%84%85%E1%85%A2%E1%86%B7%2004d2fd35f1744dda99709ca24c228252/Untitled.png)
> 

> **[예제 4-26] 컬러 히스토그램(4.26_histo_rgb.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> import matplotlib.pyplot as plt
> 
> # 이미지 읽기 및 출력
> img = cv2.imread('./img/mountain.jpg')
> 
> plt.imshow(img[:, :, ::-1])
> plt.xticks([])
> plt.yticks([])
> plt.show()
> 
> # 히스토그램 계산 및 그리기
> channels = cv2.split(img)
> colors = ('b', 'g', 'r')
> 
> for (ch, color) in zip(channels, colors):
>     hist = cv2.calcHist([ch], [0], None, [256], [0, 256])
>     plt.plot(hist, color = color)
> plt.show()
> ```
> 
> ![Untitled](4%205%20%E1%84%92%E1%85%B5%E1%84%89%E1%85%B3%E1%84%90%E1%85%A9%E1%84%80%E1%85%B3%E1%84%85%E1%85%A2%E1%86%B7%2004d2fd35f1744dda99709ca24c228252/Untitled%201.png)
> 

히스토그램을 보면 파란 하늘이 가장 넓은 영역을 차지하고 있으므로 파란색 분포가 크고 초록 나무와 단풍 때문에 초록색과 빨간색의 분포가 그 뒤를 따르는 것으로 보인다.

### 4.5.2 노멀라이즈

**노멀라이즈(normalize, 정규화)**는 기준이 서로 다른 값을 같은 기준이 되게 만드는 것을 말한다. 

절대적인 기준 대신 특정 구간으로 노멀라이즈하면 특정 부분에 몰려 있는 값을 전체 영역으로 골고루 분포하게 할 수도 있다. 이때 필요한 것이 바로 구간 노멀라이즈이다.

$$
I_N = (I - Min){new Max-newMin \over Max - Min}+newMin
$$

- $I$ : 노멀라이즈 이전 값
- $Min, Max$ : 노멀라이즈 이전 범위의 최소값, 최대값
- $newMin, newMax$ : 노멀라이즈 이후 범위의 최소값, 최대값
- $I_N$ : 노멀라이즈 이후 값

영상 분야에서는 노멀라이즈를 가지고 픽셀 값들이 0~255에 골고루 분포하지 않고 특정 영역에 몰려 있는 경우가 있다. 화질을 개선하기도 하고 영상 간 연산을 해야 하는데 서로 조건이 다른 경우 같은 조건으로 만들 수 있다.

OpenCV는 노멀라이즈 기능을 함수로 제공한다.

- dst = cv2.normalize(src, dst, alpha, beta, type_flag)
    - src : 노멀라이즈 이전 데이터
    - dst : 노멀라이즈 이후 데이터
    - alpha : 노멀라이즈 구간1
    - beta : 노멀라이즈 구간2, 구간 노멀라이즈가 아닌 경우 사용 안함
    - type_flag : 알고리즘 선택 플래그 상수
        - cv2.NORM_MINMAX : alpha와 beta 구간으로 노멀라이즈
        - cv2.NORM_L1 : 전체 합으로 나누기, alpha = 노멀라이즈 전체 합
        - cv2.NORM_L2 : 단위 벡터(unit vector)로 노멀라이즈
        - cv2.NORM_INF : 최대값으로 나누기

> **[예제 4-27] 히스토그램 정규화(4.27_hist_normalize.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> import matplotlib.pyplot as plt
> 
> # 그레이 스케일로 이미지 읽기
> img = cv2.imread('./img/abnormal.jpg', cv2.IMREAD_GRAYSCALE)
> 
> # 정규화 직접 연산
> img_f = img.astype(np.float32)
> img_norm = ((img_f - img_f.min()) * (255) / (img_f.max() - img_f.min()))
> img_norm = img_norm.astype(np.uint8)
> 
> # OpenCV API를 이용한 정규화
> img_norm2 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
> 
> # 히스토그램 계산
> hist = cv2.calcHist([img], [0], None, [256], [0, 255])
> hist_norm = cv2.calcHist([img_norm], [0], None, [256], [0, 255])
> hist_norm2 = cv2.calcHist([img_norm2], [0], None, [256], [0, 255])
> 
> # 결과 출력
> imgs = {'Before':img, 'Manual':img_norm, 'cv2.normalize()':img_norm2,
>        'Before_Hist':hist, 'Manual_Hist':hist_norm, 'cv2.normalize()_Hist':hist_norm2}
> 
> fig = plt.figure(figsize=(20, 10))
> for i, (k, v) in enumerate(imgs.items()):
>     plt.subplot(2, 3, i+1)
>     plt.title(k)
>     if i < 3:
>         plt.imshow(v, cmap='gray')
>         plt.xticks([])
>         plt.yticks([])
>     else:
>         plt.plot(v)
> plt.show()
> ```
> 
> ![Untitled](4%205%20%E1%84%92%E1%85%B5%E1%84%89%E1%85%B3%E1%84%90%E1%85%A9%E1%84%80%E1%85%B3%E1%84%85%E1%85%A2%E1%86%B7%2004d2fd35f1744dda99709ca24c228252/Untitled%202.png)
> 

구간 노멀라이즈가 아니라 서로 다른 히스토그램의 빈도를 같은 조건으로 비교하는 경우에는 전체의 비율로 노멀라이즈 해야한다.

```python
norm = cv2.normalize(hist, None, 1, 0, cv2.NORM_L1)
```

**cv2.NORM_L1** 플래그 상수를 사용하면 전체를 모두 합했을 때 1이 되는 결과가 나온다. 세 번째 인자 값에 따라 그 합은 달라지고 네 번째 인자는 무시된다.

### 4.5.3 이퀄라이즈

노멀라이즈는 분포가 한 곳에 집중되어 있는 경우에 효과적이지만, 집중된 영역에서 멀리 떨어진 값이 있을 경우에는 효과가 없다. 이때 **이퀄라이즈(equalize, 평탄화)**가 필요하다.

이퀄라이즈는 히스토그램으로 빈도를 구해서 그것을 노멀라이즈 한 후 누적값을 전체 개수로 나누어 나온 결과 값을 히스토그램 원래 픽셀 값에 매핑한다.

$$
H'(v) = round \left( {cdf(v) - cdf_{min} \over (M \times N)-cdf_{min}} \times (L - 1) \right)
$$

- $cdf(v)$ : 히스토그램 누적 함수
- $cdf_{min}$ : 누적 최소 값, 1
- $M \times N$ : 픽셀 수, 폭 X 높이
- $L$ : 분포 영역, 256
- $round(v)$ : 반올림
- $H'(v)$ : 이퀄라이즈 된 히스토그램 값

이퀄라이즈는 각각의 값이 전체 분포에 차지하는 비중에 따라 분포를 재분배하므로 명암 대비(contrast)를 개선하는데 효과적이다.

- **dst = cv2.equalizeHist(src[, dst])**
    - src : 대상 이미지, 8비트 1채널
    - dst : 결과 이미지

> **[예제 4-28] 그레이 스케일 이퀄라이즈 적용(4.28_histo_equalize.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> import matplotlib.pyplot as plt
> 
> # 이미지를 그레이 스케일로 읽기
> img = cv2.imread('./img/yate.jpg', cv2.IMREAD_GRAYSCALE)
> rows, cols = img.shape[:2]
> 
> # 원본 이미지 확인
> plt.imshow(img, cmap='gray')
> plt.xticks([])
> plt.yticks([])
> plt.show()
> 
> # 이퀄라이즈 연산을 직접 적용
> hist = cv2.calcHist([img], [0], None, [256], [0, 256])  # 히스토그램 계산
> cdf = hist.cumsum()                                     # 누적 히스토그램
> cdf_m = np.ma.masked_equal(cdf, 0)                      # 0(zero)인 값을 NaN으로 제거
> cdf_m = (cdf_m - cdf_m.min()) / (rows * cols) * 255     # 이퀄라이즈 히스토그램 계산
> cdf = np.ma.filled(cdf_m, 0).astype('uint8')            # NaN을 다시 0으로 환원
> img2 = cdf[img]                                         # 히스토그램을 픽셀로 매핑
> 
> # OpenCV API로 이퀄라이즈 히스토그램 적용
> img3 = cv2.equalizeHist(img)
> 
> # 이퀄라이즈 결과 히스토그램 계산
> hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
> hist3 = cv2.calcHist([img3], [0], None, [256], [0, 256])
> 
> # 결과 출력
> imgs = {'Before':img, 'Manual':img2, 'cv2.equalizeHist()':img3}
> hists = {'Before':hist, 'Manual':hist2, 'cv2.equalizeHist()':hist3}
> 
> fig = plt.figure(figsize=(20, 10))
> for i, (k, v) in enumerate(imgs.items()):
>     plt.subplot(2, 3, i+1)
>     plt.title(k)
>     plt.imshow(v, cmap='gray')
>     plt.xticks([])
>     plt.yticks([])
> for i, (k, v) in enumerate(hists.items()):
>     plt.subplot(2, 3, i+4)
>     plt.title(k)
>     plt.plot(v)
> plt.show()
> ```
> 
> ![Untitled](4%205%20%E1%84%92%E1%85%B5%E1%84%89%E1%85%B3%E1%84%90%E1%85%A9%E1%84%80%E1%85%B3%E1%84%85%E1%85%A2%E1%86%B7%2004d2fd35f1744dda99709ca24c228252/Untitled%203.png)
> 

코드의 길이는 다르지만 직접 연산과 OpenCV를 이용한 이퀄라이즈를 적용한 이미지의 밝기가 개선된 것을 알 수 있다. 

히스토그램 이퀄라이즈는 컬러 스케일에도 적용할 수 있다. 컬러 이미지의 밝기 값을 개선하기 위해서는 3개 채널 모두를 개선해야 하는 BGR 컬러 스페이스보다는 YUV나 HSV로 변환해서 밝기 채널만을 연산해서 최종 이미지에 적용하는 것이 좋다.

> **[예제 4-29] 컬러 이미지에 대한 이퀄라이즈 적용(4.29_histo_equalize_yuv.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> import matplotlib.pyplot as plt
> 
> # 이미지 읽기, BGR 스케일
> img = cv2.imread('./img/yate.jpg')
> 
> # 컬러 스케일을 BGR에서 YUV로 변경
> img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
> 
> # YUV 컬러 스케일의 첫 번째 채널에 대해서 이퀄라이즈 적용
> img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
> 
> # 컬러 스케일을 YUV에서 BGR로 변경
> img2 = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
> 
> # 결과 출력
> fig = plt.figure(figsize=(16, 8))
> plt.subplot(1, 2, 1)
> plt.title('Before')
> plt.imshow(img[:, :, ::-1])
> plt.xticks([])
> plt.yticks([])
> 
> plt.subplot(1, 2, 2)
> plt.title('After')
> plt.imshow(img2[:, :, ::-1])
> plt.xticks([])
> plt.yticks([])
> 
> plt.show()
> ```
> 
> ![Untitled](4%205%20%E1%84%92%E1%85%B5%E1%84%89%E1%85%B3%E1%84%90%E1%85%A9%E1%84%80%E1%85%B3%E1%84%85%E1%85%A2%E1%86%B7%2004d2fd35f1744dda99709ca24c228252/Untitled%204.png)
> 

YUV뿐만 아니라 HSV의 세 번째 채널에 대해서 이퀄라이즈를 적용해도 비슷한 결과를 얻을 수 있다.

```python
img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])
```

### 4.5.4 CLAHE

**CLAHE(Contrast Limiting Adaptive Histogram Equalization)**는 영상 전체에 이퀄라이즈를 적용했을 때 너무 밝은 부분이 날아가는 현상을 막기 위해 영상을 일정한 영역으로 나눠서 이퀄라이즈를 적용하는 것을 말한다. 노이즈가 증폭되는 것을 막기 위해 어느 히스토그램 계급(bin)이든 지정된 제한 값을 넘으면 그 픽셀은 다른 계급으로 배분하고 나서 이퀄라이즈를 적용한다.

![CLAHE 알고리즘([https://en.wikipedia.org/wiki/Adaptive_histogram_equalization#CLAHE](https://en.wikipedia.org/wiki/Adaptive_histogram_equalization#CLAHE))](4%205%20%E1%84%92%E1%85%B5%E1%84%89%E1%85%B3%E1%84%90%E1%85%A9%E1%84%80%E1%85%B3%E1%84%85%E1%85%A2%E1%86%B7%2004d2fd35f1744dda99709ca24c228252/Untitled%205.png)

CLAHE 알고리즘([https://en.wikipedia.org/wiki/Adaptive_histogram_equalization#CLAHE](https://en.wikipedia.org/wiki/Adaptive_histogram_equalization#CLAHE))

- **clahe = cv2.createCLAHE(clipLimit, tileGridSize)** : CLAHE 생성
    - clipLimit : Contrast 제한 경계 값, 기본 40.0
    - tileGridSize : 영역 크기, 기본 8 x 8
    - clahe : 생성된 CLAHE 객체
- **clahe.apply(src)** : CLAHE 적용
    - src : 입력 영상

> **[예제 4-30] CLAHE 적용(4.30_histo_clahe.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> import matplotlib.pyplot as plt
> 
> # 이미지 로드, YUV 컬러 스페이스 변경
> img = cv2.imread('./img/bright.jpg')
> img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
> 
> # 밝기 채널에 대해서 이퀄라이즈 적용
> img_eq = img_yuv.copy()
> img_eq[:, :, 0] = cv2.equalizeHist(img_eq[:, :, 0])
> img_eq = cv2.cvtColor(img_eq, cv2.COLOR_YUV2BGR)
> 
> # 밝기 채널에 대해서 CLAHE 적용
> img_clahe = img_yuv.copy()
> # CLAHE 생성
> clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
> # CLAHE 적용
> img_clahe[:, :, 0] = clahe.apply(img_clahe[:, :, 0])
> img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_YUV2BGR)
> 
> # 결과 출력
> imgs = {'Before':img, 'CLAHE':img_clahe, 'equalizeHist':img_eq}
> fig = plt.figure(figsize=(10, 5))
> for i, (k, v) in enumerate(imgs.items()):
>     plt.subplot(1, 3, i+1)
>     plt.title(k)
>     plt.imshow(v[:,:,::-1])
>     plt.xticks([])
>     plt.yticks([])
> plt.show()
> ```
> 
> ![Untitled](4%205%20%E1%84%92%E1%85%B5%E1%84%89%E1%85%B3%E1%84%90%E1%85%A9%E1%84%80%E1%85%B3%E1%84%85%E1%85%A2%E1%86%B7%2004d2fd35f1744dda99709ca24c228252/Untitled%206.png)
> 

### 4.5.5 2D 히스토그램

2차원 히스토그램은 축이 2개이고 각각의 축이 만나는 지점의 개수를 표현한다. 이를 적절히 표현하려면 3차원 그래프가 필요하다. 산을 찍은 이미지를 2차원 히스토그램으로 표현해보자.

> **[예제 4-31] 2D 히스토그램(4.31_histo_2d.ipynb)**
> 
> 
> ```python
> import cv2
> import matplotlib.pyplot as plt
> 
> # 컬러 스타일을 1.x 스타일로 사용
> plt.style.use('classic')
> img = cv2.imread('./img/mountain.jpg')
> 
> # 원본 이미지 확인
> plt.imshow(img[:,:,::-1])
> plt.xticks([])
> plt.yticks([])
> plt.show()
> 
> # 2D 히스토그램 출력
> fig = plt.figure(figsize=(16, 8))
> plt.subplot(131)
> hist = cv2.calcHist([img], [0, 1], None, [32, 32], [0, 256, 0, 256])
> p = plt.imshow(hist)
> plt.title('Blue and Green')
> plt.colorbar(p)
> 
> plt.subplot(132)
> hist = cv2.calcHist([img], [1, 2], None, [32, 32], [0, 256, 0, 256])
> p = plt.imshow(hist)
> plt.title('Green and Red')
> plt.colorbar(p)
> 
> plt.subplot(133)
> hist = cv2.calcHist([img], [0, 2], None, [32, 32], [0, 256, 0, 256])
> p = plt.imshow(hist)
> plt.title('Blue and Red')
> plt.colorbar(p)
> 
> plt.tight_layout()
> plt.show()
> ```
> 
> ![Untitled](4%205%20%E1%84%92%E1%85%B5%E1%84%89%E1%85%B3%E1%84%90%E1%85%A9%E1%84%80%E1%85%B3%E1%84%85%E1%85%A2%E1%86%B7%2004d2fd35f1744dda99709ca24c228252/Untitled%207.png)
> 
> ![Untitled](4%205%20%E1%84%92%E1%85%B5%E1%84%89%E1%85%B3%E1%84%90%E1%85%A9%E1%84%80%E1%85%B3%E1%84%85%E1%85%A2%E1%86%B7%2004d2fd35f1744dda99709ca24c228252/Untitled%208.png)
> 

2차원 히스토그램의 의미는 x축이면서 y축인 픽셀의 분포를 알 수 있다는 것이다. 논리 연산의 AND 연산과 같다.

### 4.5.6 역투영

2차원 히스토그램과 HSV 컬러 스페이스를 이용하면 색상으로 특정 물체나 사물의 일부분을 배경에서 분리할 수 있다. 물체가 있는 관심영역의 H와 V값의 분포를 얻어낸 후 전체 영상에서 해당 분포의 픽셀만 찾아내는 것이다.

> **[예제 4.32] 마우스로 선택한 영역의 물체 배경 제거[4.32_histo_backproject.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> import matplotlib.pyplot as plt
> 
> win_name = 'back_projection'
> img = cv2.imread('./img/pump_horse.jpg')
> hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
> draw = img.copy()
> 
> # 역투영된 결과를 마스킹해서 결과를 출력하는 함수
> def masking(bp, win_name):
>     disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
>     cv2.filter2D(bp, -1, disc, bp)
>     _, mask = cv2.threshold(bp, 1, 255, cv2.THRESH_BINARY)
>     result = cv2.bitwise_and(img, img, mask=mask)
>     cv2.imshow(win_name,result)
> 
> # 직접 구현한 역투영 함수
> def backProject_manual(hist_roi):
>     # 전체 영상에 대한 H, S 히스토그램 계산
>     hist_img = cv2.calcHist([hsv_img], [0, 1], None, [180, 256], [0, 180, 0, 256])
>     # 선택 영역과 전체 영상에 대한 히스토그램 비율 계산
>     hist_rate = hist_roi / (hist_img + 1)
>     # 비율에 맞는 픽셀 값 매핑
>     h, s, v = cv2.split(hsv_img)
>     bp = hist_rate[h.ravel(), s.ravel()]
>     bp = np.minimum(bp, 1)
>     bp = bp.reshape(hsv_img.shape[:2])
>     cv2.normalize(bp, bp, 0, 255, cv2.NORM_MINMAX)
>     bp = bp.astype(np.uint8)
>     # 역투영 결과로 마스킹해서 결과 출력
>     masking(bp, 'result_manual')
> 
> # OpenCV API로 구현한 함수
> def backProject_cv(hist_roi):
>     # 역투영 함수 호출
>     bp = cv2.calcBackProject([hsv_img], [0, 1], hist_roi, [0, 180, 0, 256], 1)
>     # 역투영 결과로 마스킹해서 결과 출력
>     masking(bp, 'result_cv')
> 
> # ROI 선택
> (x, y, w, h) = cv2.selectROI(win_name, img, False)
> if w > 0 and h > 0:
>     roi = img[y:y+h, x:x+w]
>     cv2.rectangle(draw, (x, y), (x+w, y+h), (0, 0, 255), 2)
>     # 선택한 ROI를 HSV 컬러 스페이스로 변경
>     hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
>     # H,S 채널에 대한 히스토그램 계산
>     hist_roi = cv2.calcHist([hsv_roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
>     # ROI의 히스토그램을 메뉴얼 구현함수와 OpenCV를 이용하는 함수에 각각 전달
>     backProject_manual(hist_roi)
>     backProject_cv(hist_roi)
>     
> cv2.imshow(win_name, draw)
> cv2.waitKey()
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled](4%205%20%E1%84%92%E1%85%B5%E1%84%89%E1%85%B3%E1%84%90%E1%85%A9%E1%84%80%E1%85%B3%E1%84%85%E1%85%A2%E1%86%B7%2004d2fd35f1744dda99709ca24c228252/Untitled%209.png)
> 

**backProject_manual** 함수에 전달된 관심영역의 히스토그램을 전체 영상의 히스토그램으로 나누어 비율을 구한다. 비율을 구한다는 것은 관심영역과 비슷한 색상 분포를 갖는 히스토그램은 1에 가까운 값을 갖고 그 반대는 0 또는 0에 가까운 값을 갖게 되는 것으로 마스킹에 사용하기 좋다. 구한 비율은 원래 영상의 H와 S 픽셀 값에 매핑한다.

여기서 **bp = hist_rate[h.ravel(), s.ravel()]** 가 핵심 코드가 된다. hist_rate는 히스토그램 비율을 값으로 가지고 있고, h와 s는 실제 영상의 각 픽셀에 해당한다. 따라서 H와 S가 교차되는 지점의 비율을 그 픽셀의 값으로 하는 1차원 배열을 얻게 된다.

이렇게 얻은 값은 비율값이기 때문에 1을 넘어서는 안되므로 np.minimim(bp, 1)로 1을 넘는 수는 1을 갖게 하고나서 1차원 배열을 원래의 shape로 만들고 0~255 그레이 스케일에 맞는 픽셀 값으로 노멀라이즈 한다. 마지막으로 연산 도중 float 타입으로 변경된 것을 uint8로 변경하면 끝나게 된다.

OpenCV에서는 함수로 이와 같은 기능을 제공한다.

- **cv2.calcBackProject(img, channel, hist, ranges, scale)**
    - img : 입력 영상, [img] 처럼 리스트로 감싸서 표현
    - channel : 처리할 채널, 리스트로 감싸서 표현
        - 1채널:[0], 2채널:[0, 1], 3채널:[0, 1, 2]
    - hist : 역투영에 사용할 히스토그램
    - ranges : 각 픽셀이 가질 수 있는 값으 범위
    - scale : 결과에 적용할 배율 계수

역투영의 장점은 알파 채널이나 크로마 키 같은 보조 역할이 없어도 복잡한 모양의 사물을 분리할 수 있다는 것이다. 하지만 대상 사물의 색상과 비슷한 색상이 섞여 있을 때는 효과가 떨어지는 단점도 있다.

### 4.5.7 히스토그램 비교

히스토그램은 영상의 픽셀 값의 분포를 갖는 정보이므로 이것을 비교하여 영상에 사용한 픽셀의 색상 비중이 얼마나 비슷한지 알 수 있다. 두 이미지가 서로 얼마나 비슷한지를 알 수 있는 하나의 방법이다.

- **cv2.compareHist(hist1, hist2, meethod)**
    - hist1, hist2 : 비교할 2개의 히스토그램, 크기와 차원이 같아야 함
    - method : 비교 알고리즘 선택 플래그 상수
        - cv2.HISTCMP_CRREL : 상관관계(1 : 완전 일치, -1 : 최대 불일치, 0 : 무관계)
            
            [Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)
            
        - cv2.HISTCMP_CHISQR : 카이제곱(0 : 완전 일치, 큰 값(미정) : 최대 불일치)
            
            [Chi-squared test](https://en.wikipedia.org/wiki/Chi-squared_test)
            
        - cv2.HISTCMP_INTERSECT : 교차(1 : 완전 일치, 0 : 최대 불일치(1로 정규화한 경우))
        - cv2.HISTCMP_BHATTACHARYYA : 바타차야(0 : 완전 일치, 1 : 최대 불일치)
            
            [Bhattacharyya distance](https://en.wikipedia.org/wiki/Bhattacharyya_distance)
            
        - cv2.HISTCMP_HELLINGER : HISTCMP_BHATTACHARYYA와 동일

서로 다른 영상의 히스토그램을 같은 조건으로 비교하기 위해서는 먼저 히스토그램을 노멀라이즈 해야한다. 이미지가 크면 픽셀 수가 많고 당연히 히스토그램의 값도 더 커지기 때문이다.

> **[예제 4-33] 히스토그램 비교(4.33_histo_compare.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> import matplotlib.pyplot as plt
> 
> # 이미지 로드
> img1 = cv2.imread('./img/taekwonv1.jpg')
> img2 = cv2.imread('./img/taekwonv2.jpg')
> img3 = cv2.imread('./img/taekwonv3.jpg')
> img4 = cv2.imread('./img/dr_ochanomizu.jpg')
> 
> # cv2.imshow('query', img1)
> imgs = [img1, img2, img3, img4]
> hists = []
> for i, img in enumerate(imgs):
>     plt.subplot(1, len(imgs), i+1)
>     plt.title(f'img{i+1}')
>     plt.axis('off')
>     plt.imshow(img[:,:,::-1])
>     
>     # 각 이미지를 HSV로 변환
>     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
>     
>     # H, S 채널에 대한 히스토그램 계산
>     hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
>     
>     # 0 ~ 1로 정규화
>     cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
>     hists.append(hist)
> 
> query = hists[0]
> methods = {'CORREL' : cv2.HISTCMP_CORREL, 'CHISQR':cv2.HISTCMP_CHISQR,
>           'INTERSECT' : cv2.HISTCMP_INTERSECT, 'BHATTACHARYYA' : cv2.HISTCMP_BHATTACHARYYA}
> for j, (name, flag) in enumerate(methods.items()):
>     print(f'{name:10s}', end='\t')
>     for i, (hist, img) in enumerate(zip(hists, imgs)):
>         # 각 메서드에 따라 img1과 각 이미지의 히스토그램 비교
>         ret = cv2.compareHist(query, hist, flag)
>         if flag == cv2.HISTCMP_INTERSECT:
>             ret = ret/np.sum(query)
>         print(f'img{i+1}:{ret:7.2f}', end='\t')
>     print()
> plt.show()
> 
> >>> CORREL        	img1:   1.00	img2:   0.70	img3:   0.56	img4:   0.23	
> 		CHISQR        	img1:   0.00	img2:  67.33	img3:  35.71	img4:1129.49	
> 		INTERSECT     	img1:   1.00	img2:   0.54	img3:   0.40	img4:   0.18	
> 		BHATTACHARYYA 	img1:   0.00	img2:   0.48	img3:   0.47	img4:   0.79
> ```
> 
> ![Untitled](4%205%20%E1%84%92%E1%85%B5%E1%84%89%E1%85%B3%E1%84%90%E1%85%A9%E1%84%80%E1%85%B3%E1%84%85%E1%85%A2%E1%86%B7%2004d2fd35f1744dda99709ca24c228252/Untitled%2010.png)
> 

img1과의 비교 결과는 모두 완전한 일치를 보여주고 있으며, img4의 경우 가장 멀어진 값으로 나타나는 것을 확인할 수 있다.
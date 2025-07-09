# 6.1 컨볼루션과 블러링

**컨볼루션(convolution) 연산**은 공간 영역 필터의 핵심이라고 할 수 있다.

블러링 실습을 통해 컨볼루션 연산이 어떻게 동작하는지 알아보자.

### 6.1.1 필터와 컨볼루션

공간 영역 필터는 연산 대상 픽셀과 그 주변 픽셀 값을 활용하는데, 주변 픽셀들 중 어디까지를 포함할 것인지 그리고 결과값을 어떻게 산출할 것인지를 결정하는 것이 **커널(kernel)**이다.

n x n 크기 커널의 각 요소와 대응하는 입력 픽셀값을 곱해서 모두 합한 것을 결과 픽셀값으로 결정하고, 이것을 마지막 픽셀까지 반복하는 것을 **컨볼루션 연산**이라고 한다.

![Untitled](https://user-images.githubusercontent.com/69300448/224931730-74d2b636-dde8-48d0-8058-349364990cc5.png)

출처: [https://www.slipp.net/wiki/pages/viewpage.action?pageId=26641520](https://www.slipp.net/wiki/pages/viewpage.action?pageId=26641520)

위 그림은 3 x 3 크기의 커널로 컨볼루션 연산을 하는 예시이다. 커널에 지정한 값의 비중에 따라 주변 요소들의 값이 반영된 새로운 영상 결과를 얻을 수 있다. 이때 커널의 크기와 값을 어떻게 하느냐에 따라 결과 영상에 필터를 적용한 효과가 달라진다.

- 주변 요소 값들의 평균 값을 반영하면 전체적인 영상은 흐릿해진다.
- 주변 요소 값들과의 차이를 반영하면 또렷해진다.

  OpenCV는 컨볼루션 연산을 담당하는 함수를 제공한다.

- **dst = cv2.filter2D(src, ddepth, kernel[, dst, anchor, delta, borderType])**
    - src : 입력 영상, Numpy 배열
    - ddepth : 출력 영상의 dtype
        - -1 : 입력 영상과 동일
        - CV_8U, CV16U/CV16S, CV_32F, CV_64F
    - kernel : 컨볼루션 커널, float32의 n x n 크기의 배열
    - dst : 결과 영상, Numpy 배열
    - anchor : 커널의 기준점, default : 중심점(-1, -1)
    - delta : 필터 적용된 결과에 추가할 값
    - borderType : 외곽 픽셀 보정 방법 지정

### 6.1.2 평균 블러링

영상을 초점이 맞지 않은 것처럼 흐릿하게 만드는 것을 **블러링(blurring)** 또는 **스무딩(smoothing)**이라고 한다. 블러링을 적용하는 가장 손쉬운 방법은 주변 픽셀 값들의 평균을 적용하는 것이다. 

평균값을 적용한다는 것은 다른 픽셀과 비슷한 값을 갖게 하는 것이므로 전체적인 영상의 픽셀값의 차이가 적어져서 이미지는 흐릿해진다.

특정 영역의 픽셀들의 평균을 구하는 것은 $1 \over n$값을 커널에 적용해서 컨볼루션 하는 것과 같다.

$$
k = {1 \over 25}
\begin{bmatrix} 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 & 1 
\\ 1 & 1 & 1 & 1 & 1 
\\ 1 & 1 & 1 & 1 & 1 
\\ 1 & 1 & 1 & 1 & 1 
\end{bmatrix}
$$

> **[예제 6-1] 평균 필터를 생성해서 블러 적용(6.1_blur_avg_kernel.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> img = cv2.imread('./img/girl.jpg')
> 
> # 5x5 평균 필터 커널 생성
> kernel = np.ones((5, 5))/5**2
> 
> # 필터 적용
> blured = cv2.filter2D(img, -1, kernel)
> 
> # 결과 출력
> cv2.imshow('origin', img)
> cv2.imshow('avrg blur', blured)
> cv2.waitKey()
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled 1](https://user-images.githubusercontent.com/69300448/224931945-1ebf3b94-149c-4b33-9790-474fba19633a.png)
> 

OpenCV는 커널을 생성하지 않고도 평균 블러를 할 수 있는 함수를 더 제공한다.

- **dst = cv2.blur(src, ksize[, dst, anchor, borderType])**
    - src : 입력 영상, Numpy 배열
    - ksize : 커널의 크기
    - 나머지 인자는 cv2.filter2D()와 동일
- **dst = cv2.boxFilter(src, ddepth, ksize[, dst, anchor, normalize, borderType])**
    - src : 입력 영상, Numpy 배열
    - ddepth : 출력 영상의 dtype, -1: 입력 영상과 동일
    - normalize : 커널 크기로 정규화($1 \over ksize^2$) 지정 여부, 불(boolean)
    - 나머지 인자는 cv2.filter2D()와 동일

**cv2.blur()** 함수는 커널의 크기만 지정하면 평균 커널을 생성해서 블러링을 적용한 영상을 만들어낸다.

**cv2.boxFilter()** 함수는 normalize 인자에 True를 지정하면 cv2.blur() 함수와 같다. False를 지정하면 커널 영역의 모든 픽셀의 합을 구하게 되는데, 이는 밀도를 이용한 객체 추적 알고리즘에서 사용한다.

> **[예제 6-2] 블러 전용 함수로 블러링 적용(6.2_blur_avg_api.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> file_name = './img/taekwonv1.jpg'
> img = cv2.imread(file_name)
> 
> # blur() 함수로 블러링
> blur1 = cv2.blur(img, (10, 10))
> 
> # boxFilter() 함수로 블러링 적용
> blur2 = cv2.boxFilter(img, -1, (10, 10))
> 
> # 결과 출력
> merged = np.hstack((img, blur1, blur2))
> cv2.imshow('blur', merged)
> cv2.waitKey()
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled 2](https://user-images.githubusercontent.com/69300448/224932018-20370574-0ffa-4090-b2e9-cee9cb22889e.png)
> 

### 6.1.3 가우시안 블러링

가우시안 분포를 갖는 커널로 블러링을 하는 것을 가우시안 블러링이라고 한다.

아래의 커널처럼 중앙값이 가장 크고 멀어질수록 그 값이 작아지는 커널을 사용하는 것을 말한다.

$$
k = {1 \over 16}
\begin{bmatrix}
1 & 2 & 1 \\
2 & 4 & 2 \\
1 & 2 & 1
\end{bmatrix}
$$

새로운 픽셀값을 선정할 때 대상 픽셀에 가까울수록 많은 영향을 주고, 멀어질수록 적은 영향을 주기 때문에 원래의 영상과 비슷하면서도 노이즈를 제거하는 효과가 있다.

OpenCV는 가우시안 분포 커널로 블러링을 적용하는 함수를 제공한다.

- **cv2.GaussianBlur(src, ksize, sigmaX[, sigmaY, borderType])**
    - src : 입력 영상
    - ksize : 커널 크기, 홀수
    - sigmaX : X 방향 표준편차
        - 0 : auto, $\sigma$ = 0.3((kisze - 1)0.5 - 1) + 0.8
    - sigmaY : Y 방향 표준편차
        - default : sigmaX
    - borderType : 외곽 테두리 보정 방식
- **ret = cv2.getGaussianKernel(ksize, sigma[, ktype])**
    - ret : 가우시안 커널(1차원이므로 ret * ret.T 형태로 사용)

> **[예제 6-3] 가우시안 블러(6.3_blur_gaussian.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> img = cv2.imread('./img/gaussian_noise.jpg')
> 
> # 가우시안 커널을 직접 생성해서 블러링
> k1 = np.array([[1, 2, 1],
>                [2, 4, 2],
>                [1, 2, 1]]) * (1/16)
> blur1 = cv2.filter2D(img, -1, k1)
> 
> # 가우시안 커널을 API로 얻어서 블러링
> k2 = cv2.getGaussianKernel(3, 0)
> blur2 = cv2.filter2D(img, -1, k2*k2.T)
> 
> # 가우시안 블러 API로 블러링
> blur3 = cv2.GaussianBlur(img, (3, 3), 0)
> 
> # 결과 출력
> print('k1:', k1)
> print('k2:', k2*k2.T)
> 
> merged = np.hstack((img, blur1, blur2, blur3))
> cv2.imshow('gaussian blur', merged)
> cv2.waitKey()
> cv2.destroyAllWindows()
> 
> >>> k1: [[0.0625 0.125  0.0625]
> 				 [0.125  0.25   0.125 ]
> 				 [0.0625 0.125  0.0625]]
> 		k2: [[0.0625 0.125  0.0625]
> 				 [0.125  0.25   0.125 ]
> 				 [0.0625 0.125  0.0625]]
> ```
> 
> ![Untitled 3](https://user-images.githubusercontent.com/69300448/224932096-a9dc00fc-4dbe-42b1-b305-7c71a70a7b7e.png)
> 

### 6.1.4 미디언 블러링

커널 영역 픽셀값 중에 중간값을 대상 픽셀의 값으로 선택하는 것을 **미디언(median) 블러링**이라고 한다. 이 필터는 기존 픽셀값 중에 하나를 선택하므로 기존값을 재활용한다는 특징이 있다. 이 필터는 **소금-후추(salt-and-pepper, 소금과 후추를 뿌린 듯한) 잡음 제거**에 효과적이다.

- **dst = cv2.medianBlur(src, ksize)**
    - src : 입력 영상, Numpy 배열
    - ksize : 커널 크기

> **[예제 6-4] 미디언 블러링(6.4_blur_median.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> img = cv2.imread('./img/salt_pepper_noise.jpg')
> 
> # 미디언 블러 적용
> blur = cv2.medianBlur(img, 5)
> 
> # 결과 출력
> merged = np.hstack((img, blur))
> cv2.imshow('median', merged)
> cv2.waitKey()
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled 4](https://user-images.githubusercontent.com/69300448/224932156-4b214414-37a1-412a-a14c-494d37e12d93.png)
> 

### 6.1.5 바이레터럴 필터

블러링 필터는 대체로 잡음을 제거하는 데 효과가 있지만, 경계도 흐릿하게 만드는 문제를 가지고있다. **바이레터럴(bilateral) 필터**는 이 문제를 개선하기 위해 가우시안 필터와 경계 필터 2개를 사용하는데, 노이즈는 없고 경계가 비교적 또렷한 영상을 얻을 수 있지만 속도가 느리다는 단점이 있다.

- **dst = cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace[, dst, borderType])**
    - src : 입력 영상, Numpy 배열
    - d : 필터의 직경(diameter), 5보다 크면 매우 느림
    - sigmaColor : 색공간 필터의 시그마값
    - sigmaSpace : 좌표 공간의 시그마 값(단순한 사용을 위해 sigmaColor와 sigmaSpace에 같은 값을 사용할 것을 권장하며, 범위는 10~150을 권장함)

> **[예제 6-5] 바이레터럴 필터와 가우시안 필터 비교(6.5_blur_bilateral.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> img = cv2.imread('./img/gaussian_noise.jpg')
> 
> # 가우시안 필터 적용
> blur1 = cv2.GaussianBlur(img, (5, 5), 0)
> 
> # 바이레터럴 필터 적용
> blur2 = cv2.bilateralFilter(img, 5, 75, 75)
> 
> # 결과 출력
> merged = np.hstack((img, blur1, blur2))
> cv2.imshow('bilateral', merged)
> cv2.waitKey()
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled 5](https://user-images.githubusercontent.com/69300448/224932200-6da3de84-7fbc-4880-be1a-c0e0a0fb0b00.png)
> 

cv2.bilateralFilter() 함수에 시그마값을 150이상 지정하면 스케치 효과를 얻을 수 있다.

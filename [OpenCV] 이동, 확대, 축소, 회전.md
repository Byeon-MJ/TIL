# 5.1 이동, 확대,축소, 회전

영상의 기하학적 변환은 기존의 영상을 원하는 모양이나 방향 등으로 변환하기 위해 각 픽셀을 새로운 위치로 옮기는 것이 작업의 대부분이다. 각 픽셀의 $x, y$ 좌표에 대해 옮기고자 하는 새로운 좌표 $x', y'$을 구하는 연산이 필요하다. 각 좌표에 대해 연산식을 적용해서 새로운 좌표를 구해야 하는데, 이때 사용할 연신삭을 가장 효과적으로 표현하는 방법이 **행렬식**이다. 변환행렬을 이용해서 영상 변환에 가장 흔하게 사용하는 이동, 확대/축소, 회전에 대해 알아보자.

### 5.1.1 이동

2차원 공간에서 물체를 다른 곳으로 이동시키려면 원래 있던 좌표에 이동시키려는 거리만큼 더해서 이동할 새로운 좌표를 구한다.

어떤 점 $p(x, y)$를 $d_x$와 $d_y$만큼 옮기면 새로운 위치의 좌표 $p(x', y')$을 구할 수 있다. 수식으로 표현하면 아래와 같다.

$$
x' = x + d_x \\
y' = y + d_y
$$

이 방적식을 행렬식으로 표현하면 아래와 같다.

$$
\begin{bmatrix} x' \\ y' \end{bmatrix} = 
\begin{bmatrix} 1 & 0 & d_x \\ 0 & 1 & d_y \end{bmatrix}
\begin{bmatrix} x \\ y \\ 1 \end{bmatrix}
$$

$$
\begin{bmatrix} x' \\ y' \end{bmatrix} =
\begin{bmatrix} x + d_x \\ y + d_y \end{bmatrix} = 
\begin{bmatrix} 1x + 0y + 1d_x \\ 0x + 1y + 1d_y \end{bmatrix}
$$

좌표를 변환하는 과정은 OpenCV가 알아서 해주지만 어떻게 변환할 것인지는 개발자가 표현해야하는데, 변환할 방정식을 함수에 전달할 때 행렬식이 표현하기 가장 적절하다. 2 x 3 변환행렬만 전달되면 연산이 가능하다. 

OpenCV는 행렬로 영상의 좌표를 변환시켜 주는 함수를 제공한다.

- **dst = cv2.warpAffine(src, mtrx, dsize [, dst, flags, dorderMode, borderValue])**
    - src : 원본 영상, Numpy 배열
    - mtrx : 2 x 3 변환 행렬, Numpy 배열, dtype = float32
    - dsize : 결과 이미지 크기, tuple(width, height)
    - flags : 보간법 알고리즘 선택 플래그
        - cv2.INTER_LINEAR : 기본 값, 인접한 4개 픽셀 값에 거리 가중치 사용
        - cv2.INTER_NEAREST : 가장 가까운 픽셀 값 사용
        - cv2.INTER_AREA : 픽셀 영역 관계를 이용한 재샘플링
        - cv2.INTER_CUBIC : 인접한 16개 픽셀 값에 거리 가중치 사용
        - cv2.INTER_LANCZOS4 : 인접한 8개 픽셀을 이용한 란초의 알고리즘
    - borderMode : 외곽 영상 보정 플래그
        - cv2.BORDER_CONSTANT : 고정 색상 값 (999 | 12345 | 999)
        - cv2.BORDER_REPLICATE : 가장자리 복제 (111 | 12345 | 555)
        - cv2.BORDER_WRAP : 반복 (345 | 12345 | 123)
        - cv2.BORDER_REFLECT : 반사 (321 | 12345 | 543)
    - borderValue : cv2.BORDER_CONSTANT의 경우 사용할 색상 값(기본 값 = 0)
    - dst : 결과 이미지, Numpy 배열

cv2.warpAffine() 함수는 src 영상을 mtrx 행렬에 따라 변환해서 dsize 크기로 만들어서 반환한다.

변환에 대부분 나타나는 픽셀 탈락 현상을 보정해 주는 보간법 알고리즘과 경계 부분의 보정 방법도 선택할 수 있다.

> **[예제 5-1] 평행 이동(5.1_translate.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> import matplotlib.pyplot as plt
> 
> img = cv2.imread('./img/fish.jpg')
> rows, cols = img.shape[:2]  # 영상의 크기
> dx, dy = 100, 50            # 이동할 픽셀 거리
> 
> # 변환 행렬 생성
> mtrx = np.float32([[1, 0, dx], [0, 1, dy]])
> 
> # 단순 이동
> dst = cv2.warpAffine(img, mtrx, (cols+dx, rows+dy))
> 
> # 탈락된 외곽 픽셀을 파란색으로 보정
> dst2 = cv2.warpAffine(img, mtrx, (cols+dx, rows+dy), None, 
>                      cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, (255, 0, 0))
> 
> # 탈락된 외곽 픽셀을 원본을 반사시켜서 보정
> dst3 = cv2.warpAffine(img, mtrx, (cols+dx, rows+dy), None,
>                      cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
> 
> # 결과 출력
> imgs = {'original':img, 'trans':dst, 'BORDER_CONSTANT':dst2, 'BORDER_REFLECT':dst3}
> 
> for i, (k, v) in enumerate(imgs.items()):
>     plt.subplot(2, 2, i+1)
>     plt.title(k)
>     plt.imshow(v[:,:,::-1])
>     plt.xticks([])
>     plt.yticks([])
> plt.show()
> ```
> 
> ![Untitled](5%201%20%E1%84%8B%E1%85%B5%E1%84%83%E1%85%A9%E1%86%BC,%20%E1%84%92%E1%85%AA%E1%86%A8%E1%84%83%E1%85%A2%20%E1%84%8E%E1%85%AE%E1%86%A8%E1%84%89%E1%85%A9,%20%E1%84%92%E1%85%AC%E1%84%8C%E1%85%A5%E1%86%AB%206e8e3407f087453ebd217f8df320c509/Untitled.png)
> 

영상 이동에는 외곽 영역 이외에는 픽셀의 탈락이 발생하지 않으므로 보간법 알고리즘을 선택하는 네 번째 인자는 의미가 없다.

### 5.1.2 확대/축소

영상을 확대 또는 축소하려면 원래 있던 좌표에 원하는 비율만큼 곱해서 새로운 좌표를 구할 수 있다. 확대/축소 비율을 가로와 세로 방향으로 각각 $\alpha$와 $\beta$라고 하면 변환행렬을 다음과 같다.

$$
\begin{bmatrix} x' \\ y' \end{bmatrix} = 
\begin{bmatrix} \alpha & 0 & 0 \\ 0 & \beta & 0 \end{bmatrix}
\begin{bmatrix} x \\ y \\ 1 \end{bmatrix}
$$

확대와 축소는 2 x 2 행렬로도 충분히 표현이 가능하지만 cv2.warpAffine() 함수는 2 x 3 행렬이 아니면 오류가 발생하기 때문에 열을 추가해서 2 x 3 행렬로 표현을 하였다.

> **[예제 5-2] 행렬을 이용한 확대와 축소(5.2_scale_matrix.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> import matplotlib.pyplot as plt
> 
> img = cv2.imread('./img/fish.jpg')
> height, width = img.shape[:2]
> 
> # 0.5배 축소 변환행렬
> m_small = np.float32([[0.5, 0, 0], [0, 0.5, 0]])
> 
> # 2배 확대 변환행렬
> m_big = np.float32([[2, 0, 0], [0, 2, 0]])
> 
> # 보간법 적용 없이 확대/축소
> dst1 = cv2.warpAffine(img, m_small, (int(height*0.5), int(width*0.5)))
> dst2 = cv2.warpAffine(img, m_big, (int(height*2), int(width*2)))
> 
> # 보간법 적용한 확대/축소
> dst3 = cv2.warpAffine(img, m_small, (int(height*0.5), int(width*0.5)), None, cv2.INTER_AREA)
> dst4 = cv2.warpAffine(img, m_big, (int(height*2), int(width*2)), None, cv2.INTER_CUBIC)
> 
> # 결과 출력
> cv2.imshow('original', img)
> cv2.imshow('small', dst1)
> cv2.imshow('big', dst2)
> cv2.imshow('small INTER_AREA', dst3)
> cv2.imshow('big INTER_CUBIC', dst4)
> cv2.waitKey(0)
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled](5%201%20%E1%84%8B%E1%85%B5%E1%84%83%E1%85%A9%E1%86%BC,%20%E1%84%92%E1%85%AA%E1%86%A8%E1%84%83%E1%85%A2%20%E1%84%8E%E1%85%AE%E1%86%A8%E1%84%89%E1%85%A9,%20%E1%84%92%E1%85%AC%E1%84%8C%E1%85%A5%E1%86%AB%206e8e3407f087453ebd217f8df320c509/Untitled%201.png)
> 

보간법 알고리즘으로는 축소에는 **cv2.INTER_AREA**가 효과적이고, 확대에는 **cv2.INTER_CUBIC**과 **cv2.INTER_LINEAR**가 효과적인 것으로 알려져 있다.

OpenCV는 변환행렬을 작성하지 않고 확대와 축소 기능을 사용하는 함수를 제공한다.

- **dst = cv2.resize(src, dsize, dst, fx, fy, interpolation)**
    - src : 입력 영상, Numpy 배열
    - dsize : 출력 영상 크기(확대/축소 목표 크기), 생략하면 fx, fy를 적용
        - (width, height)
    - fx, fy : 크기 배율, 생략하면 dsize를 적용
    - interpolation : 보간법 알고리즘 선택 플래그(cv2.warpAffine()과 동일)
    - dst : 결과 영상, Numpy 배열

cv2.resize() 함수는 확대 혹은 축소할 때 몇 픽셀로 할지 아니면 몇 퍼센트로 할지 선택할 수 있다. dsize로 변경하고 싶은 픽셀 크기를 직접 지정하거나 fx와 fy로 변경할 배율을 지정할 수 있다. dsize와 fx, fy 모두 값을 전달하면 dsize만 적용된다.

> **[예제 5-3] cv2.resize()로 확대와 축소(5.3_scale_resize.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> import matplotlib.pyplot as plt
> 
> img = cv2.imread('./img/fish.jpg')
> height, width = img.shape[:2]
> 
> # 크기 지정으로 축소
> # dst1 = cv2.resize(img, (int(width*0.5), int(height*0.5)), None, 0, 0, cv2.INTER_AREA)
> dst1 = cv2.resize(img, (int(width*0.5), int(height*0.5)), interpolation=cv2.INTER_AREA)
> 
> # 배율 지정으로 확대
> dst2 = cv2.resize(img, None, None, 2, 2, cv2.INTER_CUBIC)
> 
> # 결과 출력
> cv2.imshow('original', img)
> cv2.imshow('small', dst1)
> cv2.imshow('big', dst2)
> cv2.waitKey(0)
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled](5%201%20%E1%84%8B%E1%85%B5%E1%84%83%E1%85%A9%E1%86%BC,%20%E1%84%92%E1%85%AA%E1%86%A8%E1%84%83%E1%85%A2%20%E1%84%8E%E1%85%AE%E1%86%A8%E1%84%89%E1%85%A9,%20%E1%84%92%E1%85%AC%E1%84%8C%E1%85%A5%E1%86%AB%206e8e3407f087453ebd217f8df320c509/Untitled%202.png)
> 

### 5.1.3 회전

영상을 회전하기 위해서는 삼각함수를 사용해야 한다.

점 $p$를 원을 따라 $p'$으로 옮기는 것을 회전이라고 하고, 새로운 점 $p'$의 좌표 $x', y'$을 구해주면 회전을 할 수 있다. 

각 $\theta$만큼 회전한다고 했을 때, 새로운 점의 좌표는 $p'(xcos\theta, xsin\theta)$가 된다. 새로운 점의 사분면에 따라 다르게도 표현할 수 있다. $y$좌표를 기준으로 계산하면 $p'(-ysin\theta, ycos\theta)$로 설명할 수 있다.

행렬식으로 표현하면 아래와 같다.

$$
\begin{bmatrix} x' \\ y' \end{bmatrix} = 
\begin{bmatrix} cos\theta & -sin\theta & 0 \\ sin\theta & cos\theta & 0 \end{bmatrix}
\begin{bmatrix} x \\ y \\ 1 \end{bmatrix}
$$

> **[예제 5-4] 변환행렬로 회전(5.4_rotate_matrix.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> import matplotlib.pyplot as plt
> 
> img = cv2.imread('./img/fish.jpg')
> rows, cols = img.shape[:2]
> 
> # 라디안 각도 계산(60진법을 호도법으로 변경)
> d45 = 45.0 * np.pi / 180
> d90 = 90.0 * np.pi / 180
> 
> # 회전을 위한 변환행렬 생성
> m45 = np.float32([[np.cos(d45), -1*np.sin(d45), rows//2], 
>                  [np.sin(d45), np.cos(d45), -1*cols//4]])
> m90 = np.float32([[np.cos(d90), -1*np.sin(d90), rows],
>                  [np.sin(d90), np.cos(d90), 0]])
> 
> # 회전 변환행렬 적용
> r45 = cv2.warpAffine(img, m45, (cols, rows))
> r90 = cv2.warpAffine(img, m90, (rows, cols))
> 
> # 결과 출력
> cv2.imshow('origin', img)
> cv2.imshow('45', r45)
> cv2.imshow('90', r90)
> cv2.waitKey(0)
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled](5%201%20%E1%84%8B%E1%85%B5%E1%84%83%E1%85%A9%E1%86%BC,%20%E1%84%92%E1%85%AA%E1%86%A8%E1%84%83%E1%85%A2%20%E1%84%8E%E1%85%AE%E1%86%A8%E1%84%89%E1%85%A9,%20%E1%84%92%E1%85%AC%E1%84%8C%E1%85%A5%E1%86%AB%206e8e3407f087453ebd217f8df320c509/Untitled%203.png)
> 

회전을 위한 변환행렬 생성은 다소 까다롭고 회전 축까지 반영하려면 일이 복잡해진다. 

OpenCV에서는 복잡한 계산을 하지 않고도 **변환행렬을 생성**할 수 있는 함수를 제공한다.

- mtrx = cv2.getRotationMatrix2D(center, angle, scale)
    - center : 회전 축 중심 좌표, Tuple(x, y)
    - angle : 회전 각도, 60진법
    - scale : 확대/축소 배율

> **[예제 5-5] 회전 변환행렬 구하기(5.5_rotate_getmatrix.ipynb)**
> 
> 
> ```python
> import cv2
> 
> img = cv2.imread('./img/fish.jpg')
> rows, cols = img.shape[:2]
> 
> # 회전을 위한 변환행렬 구하기
> # 회전축 중앙, 각도 45, 배율 0.5
> m45 = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 0.5)
> # 회전축 중앙, 각도 90, 배율 1.5
> m90 = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1.5)
> 
> # 변환행렬 적용
> img45 = cv2.warpAffine(img, m45, (cols, rows))
> img90 = cv2.warpAffine(img, m90, (cols, rows))
> 
> # 결과 출력
> cv2.imshow('origin', img)
> cv2.imshow('45', img45)
> cv2.imshow('90', img90)
> cv2.waitKey(0)
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled](5%201%20%E1%84%8B%E1%85%B5%E1%84%83%E1%85%A9%E1%86%BC,%20%E1%84%92%E1%85%AA%E1%86%A8%E1%84%83%E1%85%A2%20%E1%84%8E%E1%85%AE%E1%86%A8%E1%84%89%E1%85%A9,%20%E1%84%92%E1%85%AC%E1%84%8C%E1%85%A5%E1%86%AB%206e8e3407f087453ebd217f8df320c509/Untitled%204.png)
>

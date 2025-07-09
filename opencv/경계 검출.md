# 6.2 경계 검출

영상에서 경계(edge)를 검출하는 것은 배경과 전경을 분리하는 데 가장 기본적인 작업이다. 경계 검출은 객체 인식과 추적에서 아주 중요한 작업이다.

영상의 경계를 검출해서 경계에 있는 픽셀만을 골라서 강조하면 경계를 선명하게 만들 수 있다.

### 6.2.1 기본 미분 필터

경계를 검출하려면 픽셀 값의 변화가 갑자기 크게 일어나는 지점을 찾아내야 한다. 연속된 픽셀값에 미분 연산을 통해 알 수 있다.

1차 미분 연산의 x축과 y축 방향에 대한 기본적인 커널은 다음과 같다.

$$
G_x = 
\begin{bmatrix} 
-1 & 1
\end{bmatrix}
\\ 
G_y = 
\begin{bmatrix} 
-1 \\ 1
\end{bmatrix}
$$

> **[예제 6-6] 미분 커널로 경계 검출(6.6_edge_differential.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> img = cv2.imread('./img/sudoku.jpg')
> 
> # 미분 커널 생성
> gx_kernel = np.array([[-1, 1]])
> gy_kernel = np.array([[-1], [1]])
> 
> # 필터 적용
> edge_gx = cv2.filter2D(img, -1, gx_kernel)
> edge_gy = cv2.filter2D(img, -1, gy_kernel)
> 
> # 결과 출력
> merged = np.hstack((img, edge_gx, edge_gy))
> cv2.imshow('edge', merged)
> cv2.waitKey()
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled](https://user-images.githubusercontent.com/69300448/224932413-6befd619-dbb1-4d07-8c71-153fafdc86c7.png)
> 

미분으로 얻은 엣지 정보는 각각 x축과 y축에 대한 값의 변화를 나타내는 것이고 이것을 **기울기, 그래디언트(gradient)**라고 한다.

$G_x, G_y$ 두 값을 이용하면 엣지의 강도(magnitude)와 방향(direction)이라는 정보도 추가로 얻을 수 있다. 그래디언트의 방향과 엣지의 방향은 서로 수직이다. 

$$
magnitude = \sqrt {G_x^2 + G_y^2} \\
direction(\theta) = arctan({G_y \over G_x})
$$

이 값들은 영상의 특징을 묘사하는 중요한 단서가 되고 이를 이용해 영상끼리 얼마나 비슷한지도 알아낼 수 있다.

### 6.2.2 로버츠 교차 필터

1963년 로렌스 로버츠(Lawrence Robers)가 제안한 기본 미분 커널을 개선한 커널이다.

$$
G_x = 
\begin{bmatrix} 
+1 & 0 \\
0 & -1
\end{bmatrix}
\\ 
G_y = 
\begin{bmatrix} 
0 & +1 \\
-1 & 0
\end{bmatrix}
$$

이 커널은 대각선 방향으로 1과 -1을 배치해서 사선 경계 검출 효과를 높였지만 노이즈에 민감하고 엣지 강도가 약한 단점이 있다.

> **[예제 6-7] 로버츠 마스크를 적용한 경계 검출(6.7_edge_roberts.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> img = cv2.imread('./img/sudoku.jpg')
> 
> # 로버츠 커널 생성
> gx_kernel = np.array([[1, 0], [0, -1]])
> gy_kernel = np.array([[0, 1], [-1, 0]])
> 
> # 커널 적용
> edge_gx = cv2.filter2D(img, -1, gx_kernel)
> edge_gy = cv2.filter2D(img, -1, gy_kernel)
> 
> # 결과 출력
> merged = np.hstack((img, edge_gx, edge_gy, edge_gx+edge_gy))
> cv2.imshow('roberts cross', merged)
> cv2.waitKey()
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled 1](https://user-images.githubusercontent.com/69300448/224932483-1146ea84-a6fe-440c-a1d9-ef9d079e021d.png)
> 

### 6.2.3 프리윗 필터

주디스 프리윗(Judith M. S. Prewitt)이 개발한 프리윗 마스크는 각 방향으로 차분을 세 번 계산하도록 배치해서 엣지 강도가 강하고 수직과 수평 엣지를 동등하게 찾는 장점이 있지만 대각선 검출이 약하다. 프리윗 필터는 다음과 같은 커널을 사용한다.

$$
G_x =
\begin{bmatrix}
-1 & 0 & +1 \\
-1 & 0 & +1 \\
-1 & 0 & +1
\end{bmatrix}
\\
G_y =
\begin{bmatrix}
-1 & -1 & -1 \\
0 & 0 & 0 \\
+1 & +1 & +1
\end{bmatrix}
$$

> **[예제 6-8] 프리윗 마스크를 적용한 경계 검출(6.8_edge_prewitt.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> file_name = './img/sudoku.jpg'
> img = cv2.imread(file_name)
> 
> # 프리윗 커널 생성
> gx_kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
> gy_kernel = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
> 
> # 프리윗 커널 필터 적용
> edge_gx = cv2.filter2D(img, -1, gx_kernel)
> edge_gy = cv2.filter2D(img, -1, gy_kernel)
> 
> # 결과 출력
> merged = np.hstack((img, edge_gx, edge_gy, edge_gx+edge_gy))
> cv2.imshow('prewitt cross', merged)
> cv2.waitKey()
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled 2](https://user-images.githubusercontent.com/69300448/224932530-aab14831-d1e8-46f7-85fd-ef2653bf58ac.png)
> 

### 6.2.4 소벨 필터

1968년 어윈 소벨(Irwin Sobel)이 제안한 중심 픽셀의 차분 비중을 두 배로 주어 수평, 수직, 대각선 경계 검출에 모두 강한 마스크이다.

$$
G_x =
\begin{bmatrix}
-1 & 0 & +1 \\
-2 & 0 & +2 \\
-1 & 0 & +1
\end{bmatrix}
\\
G_y =
\begin{bmatrix}
-1 & -2 & -1 \\
0 & 0 & 0 \\
+1 & +2 & +1
\end{bmatrix}
$$

소벨 마스크는 가장 대표적인 1차 미분 마스크로 OpenCV에서는 전용 함수를 제공한다.

로버츠, 프리윗 등의 필터는 실무에서는 거의 사용하지 않고 역사적인 의미와 교육적인 의미만 있다.

- **dst = cv2.Sobel(src, ddepth, dx, dy[, dst, ksize, scale, delta, borderType])**
    - src : 입력 영상, Numpy 배열
    - ddepth : 출력 영상의 dtype(-1 : 입력 영상과 동일)
    - dx, dy : 미분 차수(0, 1, 2 중 선택, 둘 다 0일수는 없음)
    - ksize : 커널의 크기(1, 3, 5, 7 중 선택)
    - scale : 미분에 사용할 계수
    - delta : 연산 결과에 가산할 값

> **[예제 6-9] 소벨 마스크를 적용한 경계 검출(6.9_edge_sobel.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> img = cv2.imread('./img/sudoku.jpg')
> 
> # Create Sobel kernel directly for edge detection
> # Create Sobel Kernel
> sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
> sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
> 
> # Apply Sobel filter
> edge_x = cv2.filter2D(img, -1, sobel_x)
> edge_y = cv2.filter2D(img, -1, sobel_y)
> 
> # Edge detection with Sobel API
> sobelx = cv2.Sobel(img, -1, 1, 0, ksize=3)
> sobely = cv2.Sobel(img, -1, 0, 1, ksize=3)
> 
> # Output Result
> merged1 = np.hstack((img, edge_x, edge_y, edge_x+edge_y))
> merged2 = np.hstack((img, sobelx, sobely, sobelx+sobely))
> merged = np.vstack((merged1, merged2))
> cv2.imshow('sobel', merged)
> cv2.waitKey()
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled 3](https://user-images.githubusercontent.com/69300448/224932636-0b569345-c37d-4e03-bca8-e9cd9112c686.png)
> 

### 6.2.5 샤르 필터

소벨 필터는 커널의 크기가 작은 경우, 또는 커널의 크기가 크더라도 그 중심에서 멀어질수록 엣지 방향성의 정확도가 떨어지는 단점이 있다. 이를 개선한 필터가 **샤르(Scharr) 필터**이다.

$$
G_x =
\begin{bmatrix}
-3 & 0 & +3 \\
-10 & 0 & +10 \\
-3 & 0 & +3
\end{bmatrix}
\\
G_y =
\begin{bmatrix}
-3 & -10 & -3 \\
0 & 0 & 0 \\
+3 & +10 & +3
\end{bmatrix}
$$

- **dst = cv2.Scharr(src, ddepth, dx, dy[, dst, scale, delta, borderType]) :**
    - 함수의 인자는 ksize가 없다는 것을 제외하면 cv2.Sobel()과 동일하다.

> **[예제 6-10] 샤르 마스크를 적용한 경계 검출(6.10_edge_scharr.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> img = cv2.imread('./img/sudoku.jpg')
> 
> # Create Scharr Kernel directly for edge detection
> gx_kernel = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
> gy_kernel = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]])
> edge_gx = cv2.filter2D(img, -1, gx_kernel)
> edge_gy = cv2.filter2D(img, -1, gy_kernel)
> 
> # Edge detection with Scharr API
> scharr_x = cv2.Scharr(img, -1, 1, 0)
> scharr_y = cv2.Scharr(img, -1, 0, 1)
> 
> # Output Result
> merged1 = np.hstack((img, edge_gx, edge_gy))
> merged2 = np.hstack((img, scharr_x, scharr_y))
> merged = np.vstack((merged1, merged2))
> cv2.imshow('Scharr', merged)
> cv2.waitKey()
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled 4](https://user-images.githubusercontent.com/69300448/224932693-449e6115-29be-461e-99d8-8d0d13aad54d.png)
> 

### 6.2.6 라플라시안 필터

미분 결과를 다시 미분하는 2차 미분을 적용하면 경계를 좀 더 확실히 검출할 수 있다. **라플라시안(Laplacian) 필터**는 대표적인 2차 미분 마스크이다.

$$
\begin{align*} {d^2f \over dy^2} 
& = {df(x, y+1) \over dy} - {df(x, y+1) \over dy} \\
& = [f(x, y+1) - f(x, y)] - [f(x, y) - f(x, y-1)] \\
& = f(x, y+1) - 2f(x, y) + f(x, y-1)
\end{align*}
$$

$$
kernel =
\begin{bmatrix}
0 & 1 & 0 \\
1 & -4 & 1 \\
0 & 1 & 0
\end{bmatrix}
$$

- **dst = cv2.Laplacian(src, ddepth[, dst, ksize, scale, delta, borderType) :**
    - 함수의 인자는 cv2.Sobel()과 동일하다.
    

> **[예제 6-11] 라플라시안 마스크를 적용한 경계 검출(6.11_edge_laplacian.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> img = cv2.imread('./img/sudoku.jpg')
> 
> # Apply Laplacian filter
> edge = cv2.Laplacian(img, -1)
> 
> # Output Result
> merged = np.hstack((img, edge))
> cv2.imshow('Laplacian', merged)
> cv2.waitKey()
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled 5](https://user-images.githubusercontent.com/69300448/224932757-56e890fc-4153-43e4-8010-6a47f98fc53f.png)
> 

라플라시안 필터는 노이즈에 민감하므로 사전에 가우시안 필터로 노이즈를 제거하고 사용하는 것이 좋다.

### 6.2.7 캐니 엣지

1986년 존 캐니(John F. Canny)가 제안한 캐니 엣지 알고리즘은 한 가지 필터를 사용하는 것이 아니라 4단계의 알고리즘을 적용한 잡음에 강한 엣지 검출기이다.

1. **노이즈 제거(Noise Reduction)** : 5 x 5 가우시안 블러링 필터로 노이즈를 제거
2. **엣지 그래디언트 방향 계산** : 소벨 마스크로 엣지 및 그래디언트 방향을 검출
3. **비최대체 억제(Non-Maximum Suppression)** : 그래디언트 방향에서 검출된 엣지 중에 가장 큰 값만 선택하고 나머지는 제거
4. **이력 스레시홀딩(Hysteresis Thresholding)** : 두 개의 경계값(Max, Min)을 지정해서 경계 영역에 있는 픽셀들 중 큰 경계값 밖의 픽셀과 연결성이 없는 픽셀을 제거

OpenCV는 이 알고리즘을 구현한 함수를 제공한다.

- **edges = cv2.Canny(img, threshold1, threshold2 [, edges, apertureSize, L2gradient])**
    - img : 입력 영상, Numpy 배열
    - threshold1, threshold2 : 이력 스레시홀딩에 사용할 최소, 최대값
    - apertureSize : 소벨 마스크에 사용할 커널 크기
    - L2gradient : 그래디언트 강도를 구할 방식 지정 플래그
        - True : $\sqrt {G_x^2 + G_y^2}$
        - False : $|G_x| + |G_y|$
    - edges : 엣지 결과값을 갖는 2차원 배열

> **[예제 6-12] 캐니 엣지 검출(6.12_edge_canny.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> img = cv2.imread('./img/sudoku.jpg')
> 
> # Apply Canny edge
> edges = cv2.Canny(img, 100, 200)
> 
> # Output Result
> cv2.imshow('Original', img)
> cv2.imshow('Canny', edges)
> cv2.waitKey()
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled 6](https://user-images.githubusercontent.com/69300448/224932797-5f23b12b-1665-402f-b981-e3d34c565c32.png)
> 

캐니 엣지는 경계 검출 결과가 뛰어나고 스레시홀드 값의 지정에 따라 경계 검출 대상을 조정할 수 있어서 가장 많이 사용되는 함수이다.

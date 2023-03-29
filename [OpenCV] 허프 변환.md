# 7.2 허프 변환

**허프 변환(Hough transform)** 은 영상에서 직선과 원 같은 간단한 모양을 식별한다. 초기에는 직선을 찾는 방법으로 시작했다가 다양한 모양을 인식하게 확장하였다.

### 7.2.1 허프 선 변환

수많은 픽셀 속에서 직선 관계를 갖는 픽셀들만 골라내는 것이 허프 선 변환의 핵심이다. 각 점마다 여러 개의 가상의 선을 그어서 그 선들 중 평면 원점과 직각을 이루는 선을 찾아 각도와 거리를 구해서 모든 점에게 동일하게 나타나는 선을 찾아주면 된다.

OpenCV는 이 선을 찾아주는 함수를 제공한다.

- **lines = cv2.HoughLines(img, rho, theta, threshold[, lines, srn=0, stn=0, min_theta, max_theta])**
    - img : 입력 영상, 1채널 바이너리 스케일
    - rho : 거리 측정 해상도, 0~1
    - theta : 각도 측정 해상도, 라디안 단위(np.pi/0~180)
    - threshold : 직선으로 판단할 최소한의 동일 개수
        - 작은 값 : 정확도 감소, 검출 개수 증가
        - 큰 값 : 정확도 증가, 검출 개수 감소
    - lines : 검출 결과, N x 1 x 2 배열($r, \theta$)
    - srn, stn : 멀티 스케일 허프 변환에 사용, 선 검출에서는 사용 안함
    - min_theta, max_theta : 검출을 위해 사용할 최대, 최소 각도

이 함수는 경계 검출한 바이너리 스케일 영상을 입력으로 전달하면 $r, \theta$를 값으로 갖는 N x 1 배열을 반환한다. 이때 거리와 각도를 얼마나 세밀하게 계산할 것인지를 rho와 theta로 전달한다.


> **[예제 7-8] 허프 선 검출(7.8_hough_line.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> img = cv2.imread('./img/sudoku.jpg')
> img2 = img.copy()
> h, w = img.shape[:2]
> 
> # Gray scale conversion and edge detection
> img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
> edges = cv2.Canny(img_gray, 100, 200)
> 
> # Hough line detection
> lines = cv2.HoughLines(edges, 1, np.pi/180, 130)
> for line in lines:
>     r, theta = line[0]    # distance and angle
>     tx, ty = np.cos(theta), np.sin(theta)    # Trigonometric ratios for x and y axes
>     x0, y0 = tx*r, ty*r    # Coordinates based on x,y axis
>     
>     # Draw red dot on based coordinate
>     cv2.circle(img2, (int(abs(x0)), int(abs(y0))), 3, (0, 0, 255), -1)
>     
>     # Calculate starting and ending points for drawing a straight line equation
>     x1, y1 = int(x0 + w*(-ty)), int(y0 + h * tx)
>     x2, y2 = int(x0 - w*(-ty)), int(y0 - h * tx)
>     
>     # Draw line
>     cv2.line(img2, (x1, y1), (x2, y2), (0, 255, 0), 1)
> 
> # Output result
> merged = np.hstack((img, img2))
> cv2.imshow('hough line', merged)
> cv2.waitKey()
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled](7%202%20%E1%84%92%E1%85%A5%E1%84%91%E1%85%B3%20%E1%84%87%E1%85%A7%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%2004a5c0f269914fa4bc5804c319a8f2a9/Untitled.png)
> 

### 7.2.2 확률적 허프 선 변환

허프 선 검출은 모든 점에 대해서 수많은 선을 그어서 직선을 찾기 때문에 많은 연산이 필요하다. 이를 개선한 것이 **점진적 확률(progressive probabilistic) 허프 변환** 이다. 이 변환은 무작위로 선정한 픽셀로 허프 변환을 수행해서 점점 그 수를 증가시켜 가는 방법이다.

OpenCV는 다음과 같은 함수를 제공한다.

- **lines = cv2.HoughLinesP(img, rho, theta, threshold[, lines, minLineLength, maxLineGap])**
    - minLineLength : 선으로 인정할 최소 길이
    - maxLineLength : 선으로 판단한 최대 간격
    - lines : 검출된 선 좌표, N x 1 x 4 배열(x1, y1, x2, y2)
    - 이외의 인자는 cv2.HoughLines()와 동일

위 함수는 cv2.HoughLines()와 거의 비슷하지만, 검출한 선의 결과값이 선의 시작과 끝 좌표라는 점과 선 검출 제약 조건으로 minLineLength, maxLineGap을 지정할 수 있다는 것이 다르다. cv2.HoughLines() 함수에 비해서 선 검출이 적게 되므로 엣지를 강하게 하고 threshold값을 낮게 지정해야 한다.


> **[예제 7-9] 확률 허프 변환으로 선 검출(7.9_hough_lineP.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> img = cv2.imread('./img/sudoku.jpg')
> img2 = img.copy()
> 
> # Gray scale conversion and edge detection
> img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
> edges = cv2.Canny(img_gray, 50, 200)
> 
> # Apply progressive probabilistic hough transform
> lines = cv2.HoughLinesP(edges, 1, np.pi/180, 10, None, 20, 2)
> for line in lines:
>     # Draw the detected line
>     x1, y1, x2, y2 = line[0]
>     cv2.line(img2, (x1, y1), (x2, y2), (0, 255, 0), 1)
> 
> merged = np.hstack((img, img2))
> cv2.imshow('Probability hough line', merged)
> cv2.waitKey()
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled](7%202%20%E1%84%92%E1%85%A5%E1%84%91%E1%85%B3%20%E1%84%87%E1%85%A7%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%2004a5c0f269914fa4bc5804c319a8f2a9/Untitled%201.png)
> 

### 7.2.3 허프 원 변환

직교좌표를 극좌표로 바꾸면 (x, y) 좌표를 ($r, \theta$) 좌표로 변환할 수 있고, 허프 직선 변환의 알고리즘을 그대로 적용해서 원을 검출할 수 있다. 

OpenCV는 메모리와 연산 속도를 이유로 이 방법보다는 캐니 엣지를 수행하고나서 소벨 필터를 적용하여 엣지의 경사도를 누적하는 방법으로 구현한다.

- **circles = cv2.HoughCircles(img, method, dp, minDist[, circles, param1, param2, minRadius, maxRadius])**
    - img : 입력 영상, 1채널 배열
    - method : 검출 방식 선택, 현재 cv2.HOUGH_GRADIENT만 가능
        - cv2.HOUGH_STANDARD
        - cv2.HOUGH_PROBABILISTIC
        - cv2.HOUGH_MULTI_SCALE
        - cv2.HOUGH_GRADIENT
    - dp : 입력 영상과 경사 누적의 해상도 반비례율, 1 : 입력과 동일, 값이 커질수록 부정확
    - minDist : 원들 중심 간의 최소 거리, 0 : 에러(동심원 검출 불가)
    - circles : 검출 원 결과, N x 1 x 3 부동 소수점 배열(x, y, radius)
    - param1 : 캐니 엣지에 전달할 스레시홀드 최대값(최소값은 최대값의 2배 작은 값을 전달)
    - param2 : 경사도 누적 경계값(값이 작을수록 잘못된 원 검출)
    - minRadius, maxRadius : 원의 최소 반지름, 최대 반지름(0이면 영상의 크기)
    

> **[예제 7-10] 허프 원 검출(7.10_hough_circle.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> img = cv2.imread('./img/coins_spread1.jpg')
> 
> # Convert to gray scale
> gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
> 
> # Gaussian blurring to remove noise
> blur = cv2.GaussianBlur(gray, (3, 3), 0)
> 
> # Apply hough circle transform(dp=1.5, minDist=30, cany_max=200)
> circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1.5, 30, None, 200)
> if circles is not None:
>     circles = np.uint16(np.around(circles))
>     for i in circles[0, :]:
>         # Draw green circles around the circumference of the circle
>         cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
>         # Draw red dot at the center of the circle
>         cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 5)
> 
> # Output result
> cv2.imshow('hough circle', img)
> cv2.waitKey()
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled](7%202%20%E1%84%92%E1%85%A5%E1%84%91%E1%85%B3%20%E1%84%87%E1%85%A7%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%2004a5c0f269914fa4bc5804c319a8f2a9/Untitled%202.png)
> 

동전 사진에서 원을 검출해보았다. 이 함수는 내부적으로 캐니 엣지를 사용하기때문에 노이즈 제거를 위해서 가우시안 블러를 처리해주었다. 캐니 엣지에서 사용할 최대 스레시홀드 값은 200으로 전달했다. dp는 원본 영상과 경사도 누적도의 해상도를 조정하는데, 1로 하면 해상도가 동일하므로 가장 정확하고, 값이 커질수록 부정확한 원을 검출할 수 있다. 1부터 시작해서 조금씩 늘려서 찾아야 한다. minDist 값은 중심점 간의 최소 거리를 의미하는 것으로 0이 들어갈 수는 없다.

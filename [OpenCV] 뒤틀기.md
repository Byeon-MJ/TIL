# 5.2 뒤틀기

이동, 확대/축소, 회전은 변환 후에도 기존의 모양이 유지되지만, **뒤틀기(warping, 워핑)**은 기존의 모양과 달라진다.

### 5.2.1 어핀 변환

어핀 변환(affine transform)은 이동, 확대/축소, 회전을 포함하는 변환으로 직선, 길이의 비율, 평행성을 보존하는 변환을 말한다.

변환 전과 후의 3개의 점을 짝 지어 매핑할 수 있다면 변환행렬을 거꾸로 계산할 수 있다.

OpenCV는 아래 함수로 이 기능을 제공한다.

- matrix = cv2.getAffineTransform(pts1, pts2)
    - pts1 : 변환 전 영상의 좌표 3개, 3 x 2 Numpy 배열(float32)
    - pts2 : 변환 후 영상의 좌표 3개, pts1과 동일
    - matrix : 변환행렬 반환, 2 x 3 행렬

> **[예제 5-6] 어핀 변환(5.6_getAffine.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> import matplotlib.pyplot as plt
> 
> file_name = './img/fish.jpg'
> img = cv2.imread(file_name)
> rows, cols = img.shape[:2]
> 
> # 변환 전, 후 각 3개의 좌표 생성
> pts1 = np.float32([[100, 50], [200, 50], [100, 200]])
> pts2 = np.float32([[80, 70], [210, 60], [250, 120]])
> 
> # 변환 전 좌표를 이미지에 표시
> cv2.circle(img, (100, 50), 5, (255, 0), -1)
> cv2.circle(img, (200, 50), 5, (0, 255, 0), -1)
> cv2.circle(img, (100, 200), 5, (0, 0, 255), -1)
> 
> # 짝지은 3개의 좌표로 변환행렬 계산
> mtrx = cv2.getAffineTransform(pts1, pts2)
> 
> # 어핀 변환 적용
> dst = cv2.warpAffine(img, mtrx, (int(cols*1.5), rows))
> 
> # 결과 출력
> cv2.imshow('origin', img)
> cv2.imshow('affine', dst)
> cv2.waitKey(0)
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled](5%202%20%E1%84%83%E1%85%B1%E1%84%90%E1%85%B3%E1%86%AF%E1%84%80%E1%85%B5%20b86ee5a1a3d54e408e4cdd2096beb3e4/Untitled.png)
> 

### 5.2.2 원근 변환

**원근 변환(perspective transform)**은 먼 것은 작게, 가까운 것은 크게 보이는 원근감을 주는 변환을 말한다. 

우리가 원근감을 느끼는 이유는 실제 세상이 3차원 좌표계이기 때문인데, 영상은 2차원 좌표계이다. 그래서 차원 간의 차이를 보정해 줄 추가 연산과 시스템이 필요하다. 이때 사용하는 좌표계를 **동차 좌표(homogeneous coordinates)**라고 한다. 이 때문에 원근 변환을 다른 말로 **호모그래피(homography)**라고도 한다.

2차원 좌표 $(x, y)$에 대응하는 동차 좌표는 기존 차수에 1개의 상수항을 추가해서 $(wx, wy, w)$로 표현하고, 이것을 2차원 좌표로 바꿀 때는 다시 상수항 $w$로 나누어 $({x \over w}, {y \over w})$로 표현한다.

원근 변환을 하려면 $(x, y, 1)$꼴의 좌표계가 필요하고, 아래와 같은 3 x 3 변환행렬식이 필요하다.

$$
w \begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = 
\begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix}
\begin{bmatrix} x \\ y \\ 1 \end{bmatrix}
$$

OpenCV는 변환 전과 후를 짝짓는 4개의 매핑 좌표를 지정해 주면 원근 변환에 필요한 3 x 3 변환행렬을 계산해주는 함수를 제공한다.

- mtrx = cv2.getPerspectiveTransform(pts1, pts2)
    - pts1 : 변환 이전 영상의 좌표 4개, 4 x 2 Numpy 배열(float32)
    - pts2 : 변환 이후 영상의 좌표 4개, pts1과 동일
    - mtrx : 변환행렬 반환, 3 x 3 행렬

> **[예제 5-7] 원근 변환(5.7_perspective.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> file_name = './img/fish.jpg'
> img = cv2.imread(file_name)
> rows, cols = img.shape[:2]
> 
> # 원근 변환 전후 4개 좌표
> pts1 = np.float32([[0, 0], [0, rows], [cols, 0], [cols, rows]])
> pts2 = np.float32([[100, 50], [10, rows-50], [cols-100, 50], [cols-10, rows-50]])
> 
> # 변환 전 좌표를 원본 이미지에 표시
> cv2.circle(img, (0, 0), 10, (255, 0, 0), -1)
> cv2.circle(img, (0, rows), 10, (0, 255, 0), -1)
> cv2.circle(img, (cols, 0), 10, (0, 0, 255), -1)
> cv2.circle(img, (cols, rows), 10, (0, 255, 255), -1)
> 
> # 원근 변환행렬 계산
> mtrx = cv2.getPerspectiveTransform(pts1, pts2)
> 
> # 원근 변환 적용
> dst = cv2.warpPerspective(img, mtrx, (cols, rows))
> 
> # 결과 출력
> cv2.imshow('origin', img)
> cv2.imshow('perspective', dst)
> cv2.waitKey(0)
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled](5%202%20%E1%84%83%E1%85%B1%E1%84%90%E1%85%B3%E1%86%AF%E1%84%80%E1%85%B5%20b86ee5a1a3d54e408e4cdd2096beb3e4/Untitled%201.png)
> 

실행 결과를 보면 원래의 이미지에서 좌표의 폭이 좁아진 부분은 작게 표현해서 마치 멀리 있는 것처럼 표현하는 것을 알 수 있다.

실제로는 그 반대로 활용하는 경우가 더 많다. 카메라로 명함이나 문서같은 것을 찍은 사진을 스캔한 것처럼 만들고 싶을 때는 반대로 원근감을 제거해야 한다.

> **[예제 5-8] 마우스와 원근 변환으로 문서 스캔 효과 내기(5.8_perspective_scan.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> win_name = 'scanning'
> img = cv2.imread('./img/paper.jpg')
> rows, cols = img.shape[:2]
> draw = img.copy()
> pts_cnt = 0
> pts = np.zeros((4, 2), dtype=np.float32)
> 
> # 마우스 이벤트 콜백 함수 구현
> def onMouse(event, x, y, flags, param):
>     global pts_cnt                           # 마우스 좌표 저장
>     if event == cv2.EVENT_LBUTTONDOWN:
>         cv2.circle(draw, (x, y), 10, (0, 255, 0), -1)  # 죄표에 초록색 동그라미 표시
>         cv2.imshow(win_name, draw)
>         
>         pts[pts_cnt] = [x, y]                 # 마우스 좌표 저장
>         pts_cnt += 1
>         
>         # 좌표가 4개 수집됨
>         if pts_cnt == 4:
>             # 죄표 4개 중 상하좌우 찾기
>             sm = pts.sum(axis=1)              # 4쌍의 좌표 각각 x+y 계산
>             diff = np.diff(pts, axis = 1)     # 4쌍의 좌표 각각 x-y 계산
>             
>             topLeft = pts[np.argmin(sm)]      # x+y가 가장 작은 값이 좌상단 좌표
>             bottomRight = pts[np.argmax(sm)]  # x+y가 가장 큰 값이 우하단 좌표
>             topRight = pts[np.argmin(diff)]   # x-y가 가장 작은 값이 우상단 좌표
>             bottomLeft = pts[np.argmax(diff)] # x-y가 가장 큰 값이 좌하단 좌표
>             
>             # 변환 전 4개의 좌표
>             pts1 = np.float32([topLeft, topRight, bottomRight, bottomLeft])
>             
>             # 변환 후 영상에 사용할 서류의 폭과 높이 계산
>             w1 = abs(bottomRight[0] - bottomLeft[0])    # 하단 좌우 좌표 거리
>             w2 = abs(topRight[0] - topLeft[0])          # 상단 좌우 좌표 거리
>             h1 = abs(topRight[1] - bottomRight[1])      # 우측 상하 좌표 거리
>             h2 = abs(topLeft[1] - bottomLeft[1])        # 좌측 상하 좌표 거리
>             width = max([w1, w2])                       # 두 좌우 거리 간의 최대 값이 서류의 폭
>             height = max([h1, h2])                      # 두 상하 거리 간의 최대 값이 서류의 높이
>             
>             # 변환 후 4개의 좌표
>             pts2 = np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])
>             
>             # 변환행렬 계산
>             mtrx = cv2.getPerspectiveTransform(pts1, pts2)
>             
>             # 원근 변환 적용
>             result = cv2.warpPerspective(img, mtrx, (int(width), int(height)))
>             
>             cv2.imshow('scanned', result)
> 
> # 결과 출력
> cv2.imshow(win_name, img)
> cv2.setMouseCallback(win_name, onMouse)
> cv2.waitKey(0)
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled](5%202%20%E1%84%83%E1%85%B1%E1%84%90%E1%85%B3%E1%86%AF%E1%84%80%E1%85%B5%20b86ee5a1a3d54e408e4cdd2096beb3e4/Untitled%202.png)
> 

나중에 배울 **“영상 분할”**과 함께 사용하면 직접 마우스로 문서 영역을 지정하지 않고 자동화 하는 방법을 알 수 있다.

### 5.2.3 삼각형 어핀 변환

어떤 영역을 여러 개의 삼각형으로 나누는 기법을 **들로네 삼각분할(Delaunay Triangulation)**이라고 한다. 영상 분야에서는 입체적 표현이마 **모핑(morphing)** 기술에 사용한다.

모핑 기술은 하나의 물체가 다른 물체로 자연스럽게 변하게 하는 것인데, 두 영상을 각각 여러 개의 삼각형으로 나누어 한 영상의 삼각형들의 크기와 모양이 나머지 영상에 대응하는 삼각형과 같아질 때까지 조금씩 바꿔서 전체적으로 하나의 영상이 다른 영상으로 자연스럽게 변하게 하는 것이다.

삼각 분할된 영역을 변환하기는 쉽지 않다. OpenCV가 제공하는 기하학적 변환 기능은 영상을 대상으로 하므로 그 대상은 늘 사각형일 수밖에 없기 떄문이다. 그래서 삼각형 모양의 변환을 하려면 다음과 같은 과정을 거쳐야 한다.

1. 변환 전 삼각형 좌표 3쌍을 정한다.
2. 변환 후 삼각형 좌표 3쌍을 정한다.
3. 과정 1의 삼각형 좌표를 완전히 감싸는 외접 사각형 좌표를 구한다.
4. 과정 3의 사각형 영역을 관심영역으로 지정한다.
5. 과정 4의 관심영역을 대상으로 과정 1과 과정 2의 좌표로 변환행렬을 구하여 어핀 변환한다.
6. 과정 5의 변환된 관심영역에서 과정2의 삼각형 좌표만 마스킹한다.
7. 과정 6의 마스크를 이용해서 원본 또는 다른 영상에 합성한다.

과정 3에서 삼각형 좌표를 완전히 감싸는 사각형의 좌표를 구하려면 **cv2.boundingRect()** 함수를 이용한다. 삼각형 뿐 아니라 다각형의 좌표를 전달하면 정확히 감싸는 외접 사각형의 좌표를 반환한다.

- **x, y, w, h = cv2.boundingRect(pts)**
    - pts : 다각형 좌표
    - x, y, w, h : 외접 사각형의 좌표와 폭과 높이

과정 6에서 삼각형 마스크를 생성하기 위해서는 **cv2.fillConvexPoly()** 함수를 쓰면 편리하다. 이 함수에 좌표를 전달하면 그 좌표 안을 원하는 색상값으로 채워주는데, 255나 0을 채우면 마스크를 쉽게 만들 수 있다.

- **cv2.fillConvexPoly(img, points, color [, lineType])**
    - img : 입력 영상
    - points : 다각형 꼭짓점 좌표
    - color : 채우기에 사용할 색상
    - lineType : 선 그리기 알고리즘 선택 플래그

> **[예제 5-9] 삼각형 어핀 변환(5.9_triangle_affine.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> img = cv2.imread('./img/taekwonv1.jpg')
> img2 = img.copy()
> draw = img.copy()
> 
> # 변환 전, 후 삼각형 좌표
> pts1 = np.float32([[188, 14], [85, 202], [294, 216]])
> pts2 = np.float32([[128, 40], [85, 307], [306, 167]])
> 
> # 각 삼각형을 완전히 감싸는 사각형 좌표 구하기
> x1, y1, w1, h1 = cv2.boundingRect(pts1)
> x2, y2, w2, h2 = cv2.boundingRect(pts2)
> 
> # 사각형을 이용한 관심영역 설정
> roi1 = img[y1:y1+h1, x1:x1+w1]
> roi2 = img2[y2:y2+h2, x2:x2+w2]
> 
> # 관심영역을 기준으로 좌표 계산
> offset1 = np.zeros((3, 2), dtype=np.float32)
> offset2 = np.zeros((3, 2), dtype=np.float32)
> for i in range(3):
>     offset1[i][0], offset1[i][1] = pts1[i][0]-x1, pts1[i][1]-y1
>     offset2[i][0], offset2[i][1] = pts2[i][0]-x2, pts2[i][1]-y2
> 
> # 관심영역을 주어진 삼각형 좌표로 어핀 변환
> mtrx = cv2.getAffineTransform(offset1, offset2)
> warped = cv2.warpAffine(roi1, mtrx, (w2, h2), None, cv2.INTER_LINEAR, cv2.BORDER_REFLECT_101)
> 
> # 어핀 변환 후 삼각형만 골라 내기 위한 마스크 생성
> mask = np.zeros((h2, w2), dtype = np.uint8)
> cv2.fillConvexPoly(mask, np.int32(offset2), (255))
> 
> # 삼각형 영역만 마스킹해서 합성
> warped_masked = cv2.bitwise_and(warped, warped, mask=mask)
> roi2_masked = cv2.bitwise_and(roi2, roi2, mask=cv2.bitwise_not(mask))
> roi2_masked = roi2_masked + warped_masked
> img2[y2:y2+h2, x2:x2+w2] = roi2_masked
> 
> # 관심영역과 삼각형에 선을 그려서 출력
> cv2.rectangle(draw, (x1, y1), (x1+w1, y1+h1), (0, 255, 0), 1)
> cv2.polylines(draw, [pts1.astype(np.int32)], True, (255, 0, 0), 1)
> cv2.rectangle(img2, (x2, y2), (x2+w2, y2+h2), (0, 255, 0), 1)
> cv2.imshow('origin', draw)
> cv2.imshow('warped triangle', img2)
> cv2.waitKey(0)
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled](5%202%20%E1%84%83%E1%85%B1%E1%84%90%E1%85%B3%E1%86%AF%E1%84%80%E1%85%B5%20b86ee5a1a3d54e408e4cdd2096beb3e4/Untitled%203.png)
>

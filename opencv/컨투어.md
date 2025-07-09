# 7.1 컨투어

컨투어(contour)는 우리말로 등고선, 윤각선, 외곽선 등을 말한다. 영상에서는 같은 색상이나 밝기의 연속된 점을 찾아 잇는 곡선을 찾아내면 모양 분석과 객체 인식에 사용할 수 있다.

- **contours, hierarchy = cv2.findContours(src, mode, method [, contours, hierarchy, offset])[-2:]**
    - src : 입력 이미지, 바이너리 스케일, 검은색 배경 흰색 전경
    - mode : 컨투어 제공 방식 선택
        - cv2.RETR_EXTERNAL : 가장 바깥쪽 라인만 제공
        - cv2.RETR_LIST : 모든 라인을 계층 없이 제공
        - c2.RETR_CCOMP : 모든 라인을 2계층으로 제공
        - cv2.RETR_TREE : 모든 라인의 모든 계층 정보를 트리 구조로 제공
    - method : 근사값 방식 선택
        - cv2.CHAIN_APPROX_NONE : 근사 계산하지 않고 모든 좌표 제공
        - cv2.CHAIN_APPROX_SIMPLE : 컨투어 꼭짓점 좌표만 제공
        - cv2.CHAIN_APPROX_TC89_L1 : Teh_Chin 알고리즘으로 좌표 개수 축소
        - cv2.CHAIN_APPROX_TC89_KCOS : Teh_Chin 알고리즘으로 좌표 개수 축소
    - contours : 검출한 컨투어 좌표, 파이썬 리스트
    - hierarchy : 컨투어 계층 정보
        - Next, Prev, FirstChild, Parent
            - -1 : 해당 사항 없음
    - offset : ROI 등으로 인해 이동한 컨투어 좌표의 오프셋
- **cv2.drawContours(img, contours, contourIdx, color, thickness) :** 컨투어 연결선 그리는 함수
    - img : 입력 영상
    - contours : 그림 그릴 컨투어 배열
    - contourIdx : 그림 그릴 컨투어 인덱스, -1 : 모든 컨투어 표시
    - color : 색상값
    - thickness : 선 두께, 0 : 채우기

cv2.findContours() 함수의 반환값은 OpenCV의 버전에 따라 달라서 버전 간에 호환되는 코드를 작성하기 위해 함수의 반환값 중 마지막 2개만 사용하도록 하는 [-2:]를 추가하는 것이 좋다.

> **[예제 7-1] 컨투어 찾기와 그리기(7.1_cntr_find.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> img = cv2.imread('./img/shapes.png')
> img2 = img.copy()
> 
> # 그레이 스케일로 변환
> img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
> 
> # 스레시홀드로 바이너리 이미지로 만들어서 검은색 배경에 흰색 전경으로 반전
> ret, imthres = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
> 
> # 가장 바깥쪽 컨투어에 대해 모든 좌표 반환
> contour, hierarchy = cv2.findContours(imthres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
> 
> # 가장 바깥쪽 컨투어에 대해 꼭짓점 좌표만 반환
> contour2, hierarchy = cv2.findContours(imthres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
> 
> # 각 컨투어의 개수 출력
> print(f'도형의 개수 : {len(contour)}({len(contour2)})')
> 
> >>> 도형의 개수 : 3(3)
> 
> # 모든 좌표를 갖는 컨투어 그리기, 초록색
> cv2.drawContours(img, contour, -1, (0, 255, 0), 4)
> 
> # 꼭짓점 좌표만을 갖는 컨투어 그리기, 초록색
> cv2.drawContours(img2, contour2, -1, (0, 255, 0), 4)
> 
> # 컨투어의 모든 좌표를 작은 파란색 점(원)으로 표시
> for i in contour:
>     for j in i:
>         cv2.circle(img, tuple(j[0]), 1, (255, 0, 0), -1)
> 
> # 컨투어의 꼭짓점 좌표를 작은 파란색 점(원)으로 표시
> for i in contour2:
>     for j in i:
>         cv2.circle(img2, tuple(j[0]), 1, (255, 0, 0), -1)
> 
> # 결과 출력
> cv2.imshow('CHAIN_APPROX_NONE', img)
> cv2.imshow('CHAIN_APPROX_SIMPLE', img2)
> cv2.waitKey()
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled](https://user-images.githubusercontent.com/69300448/226506353-a6cb3887-234c-4636-a0e0-402a8db5ac57.png)
> 

> **[예제 7-2] 컨투어 계층 트리(7.2_cntr_hierarchy.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> # 영상 읽기
> img = cv2.imread('./img/shapes_donut.png')
> img2 = img.copy()
> 
> # 바이너리 이미지로 변환
> img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
> ret, imthres = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
> 
> # 가장 바깥 컨투어만 수집
> contour, hierarchy = cv2.findContours(imthres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
> 
> # 컨투어 개수와 계층 트리 출력
> print(len(contour), hierarchy)
> 
> >>> 3 [[[ 1 -1 -1 -1]
> 		  [ 2  0 -1 -1]
> 		  [-1  1 -1 -1]]]
> 
> # 모든 컨투어를 트리 계층으로 수집
> contour2, hierarchy2 = cv2.findContours(imthres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
> 
> # 컨투어 개수와 계층 트리 출력
> print(len(contour2), hierarchy2)
> 
> >>> 6 [[[ 2 -1  1 -1]
> 		  [-1 -1 -1  0]
> 		  [ 4  0  3 -1]
> 		  [-1 -1 -1  2]
> 		  [-1  2  5 -1]
> 		  [-1 -1 -1  4]]]
> 
> # 가장 바깥 컨투어만 그리기
> cv2.drawContours(img, contour, -1, (0, 255, 0), 3)
> 
> # 모든 컨투어 그리기
> for idx, cont in enumerate(contour2):
>     # 랜덤한 컬러 추출
>     color = [int(i) for i in np.random.randint(0, 255, 3)]
>     # 컨투어 인덱스마다 랜덤한 색상으로 그리기
>     cv2.drawContours(img2, contour2, idx, color, 3)
>     # 컨투어 첫 좌표에 인덱스 숫자 표시
>     cv2.putText(img2, str(idx), tuple(cont[0][0]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
> 
> # 화면 출력
> cv2.imshow('RETR_EXTERNAL', img)
> cv2.imshow('RETR_TREE', img2)
> cv2.waitKey()
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled 1](https://user-images.githubusercontent.com/69300448/226506397-a4b66fdf-948d-4406-a6f8-2646b9a63dcf.png)
> 

### 7.1.1 이미지 모멘트와 컨투어 속성

**모멘트(moment)** 는 영상에서 대상 물체의 양적인 속성을 표현할 때 사용하는 용어이다. 이미지 모멘트는 컨투어를 이용해서 아래와 같은 공식으로 구한다.

$$
m_{p, q} = \sum_{x} \sum_{y} f(x, y) x^p y^q
$$

위 모멘트 계산 공식은 컨투어가 둘러싸는 영역의 $x, y$ 좌표의 픽셀값과 좌표 인덱스의 $p, q$ 차수를 곱한 것의 합을 구한다. 각 픽셀의 값은 바이너리 이미지이므로 0이 아닌 모든 값은 1로 계산하고 $p, q$의 차수는 0~3까지로 한다.

0~3 차수 모멘트는 **공간 모멘트** 라고 하며, 위치나 크기가 달라지면 그 값도 달라진다. 

위치가 변해도 값이 동일한 모멘트를 **중심 모멘트** 라고 하고, 아래와 같은 식으로 계산한다.

$$
\mu_{p, q} = \sum_{x} \sum_{y} f(x, y)(x - \bar{x})^p (y - \bar{y})^q
$$

- $\bar{x} = {m_{10} \over m_{00}}$
- $\bar{y} = {m_{01} \over m_{00}}$

중심 모멘트를 정규화하면 크기가 변해도 같은 값을 얻을 수 있고 공식은 아래와 같다.

$$
\nu_{p, q} = { \mu_{p, q} \over m_{00}^{(1+{p+q \over 2})} }
$$

- **moment = cv2.moments(contour)**
    - contour : 모멘트 계산 대상 컨투어 좌표
    - moment : 결과 모멘트, 파이썬 딕셔너리
        - m00, m01, m10, m11, m02, m12, m20, m21, m03, m30 : 공간 모멘트
        - mu20, mu11, mu02, mu30, mu21, mu12, mu03 : 중심 모멘트
        - nu20, nu11, nu02, nu30, nu21, nu03 : 정규화 중심 모멘트

OpenCV는 넓이와 둘레 길이 정보를 얻는 함수도 제공한다.

- **retval = cv2.contourArea(contour[, oriented=False])** : 컨투어로 넓이 계산
    - contour : 넓이를 계산할 컨투어
    - oriented : 컨투어 방향 플래그
        - True : 컨투어 방향에 따라 음수 반환
        - False : 절대값 반환
    - retval : 컨투어 영역의 넓이값
- **retval = cv2.arcLength(curve, closed)** : 컨투어로 둘레의 길이 계산
    - curve : 둘레 길이를 계산할 컨투어
    - closed : 닫힌 호인지 여부 플래그
    - retval : 컨투어의 둘레 길이 값

> **[예제 7-3] 모멘트를 이용한 중심점, 넓이, 둘레길이(7.3_contr_moment.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> img = cv2.imread('./img/shapes.png')
> 
> # 그레이 스케일 변환
> img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
> 
> # 바이너리 스케일 변환
> ret, th = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
> 
> # 컨투어 찾기
> contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
> 
> # 각 도형의 컨투어에 대한 루프
> for c in contours:
>     # 모멘트 계산
>     mmt = cv2.moments(c)
>     
>     # m10/m00, m01/m00 중심점 계산
>     cx = int(mmt['m10']/mmt['m00'])
>     cy = int(mmt['m01']/mmt['m00'])
>     
>     # 영역 넓이
>     a = mmt['m00']
>     
>     # 영역의 외곽선 길이
>     l = cv2.arcLength(c, True)
>     
>     # 중심점에 노란색 점 그리기
>     cv2.circle(img, (cx, cy), 5, (0, 255, 255), -1)
>     
>     # 중심점 근처에 넓이 그리기
>     cv2.putText(img, f'A:{a:.0f}', (cx, cy+20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
>     
>     # 컨투어 시작점에 길이 그리기
>     cv2.putText(img, f'L:{l:.2f}', tuple(c[0][0]), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))
>     
>     # 함수로 컨투어 넓이 계산해서 출력
>     print(f'area:{cv2.contourArea(c, False):.2f}')
> 
> >>> area:9870.00
> 		area:12544.00
> 		area:6216.00
> 
> # 결과 출력
> cv2.imshow('center', img)
> cv2.waitKey()
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled 2](https://user-images.githubusercontent.com/69300448/226506440-f81a64b6-fea5-45de-9e9d-4cf3f38e3ce5.png)
> 

OpenCV는 컨투어를 이용해서 해당 영역을 감싸는 여러 가지 도형 좌표를 계산하는 함수를 제공한다.

- **x, y, w, h = cv2.boundingRect(contour)** : 좌표를 감싸는 사각형 구하기
    - x, y : 사각형 왼쪽 상단 좌표
    - w, h : 폭, 높이
- **rotateRect = cv2.minAreaRect(contour)** : 좌표를 감싸는 최소한의 사각형 계산
    - rotateRect : 회전한 사각형 좌표
        - center : 중심점(x, y)
        - size : 크기(w, h)
        - angle : 회전 각(양수 : 시계 방향, 음수 : 반시계 방향)
- **vertex = cv2.boxPoints(rotateRect)** : rotateRect로부터 꼭짓점 좌표 계산
    - vertex : 4개의 꼭짓점 좌표, 소수점 포함, 정수 변환 필요
- **center, radius = cv2.minEnclosingCircle(contour)** : 좌표를 감싸는 최소한의 동그라미 계산
    - center : 원점 좌표(x, y), 튜플
    - radius : 반지름

- **area, triangle = cv2.minEnclosingTriangle(points)** : 좌표를 감싸는 최소한의 삼각형 계산
    - area : 넓기
    - triangle : 3개의 꼭짓점 좌표
- **ellipse = cv2.fitEllipse(points)** : 좌표를 감싸는 최소한의 타원 계산
    - ellipse
        - center : 원점 좌표(x, y), 튜플
        - axes : 축의 길이(x축, y축), 튜플
        - angle : 회전 각도
- **line = cv2.fitLine(points, distType, param, reps, aeps[, line])** : 중심점을 통과하는 직선 계산
    - distType : 거리 계산 방식
        - cv2.DIST_L2, cv2.DIST_L1, cv2.DIST_L12, cv2.DIST_FAIR, cv2.DIST_WELSCH, cv2.DIST_HUBER
    - param : distType에 전달할 인자, 0 = 최적값 선택
    - reps : 반지름 정확도, 선과 원본 좌표의 거리, 0.01 권장
    - aeps : 각도 정확도, 0.01 권장
    - line
        - vx, vy : 정규화된 단위 벡터, $vy \over vx$ : 직선의 기울기, 튜플
        - x0, y0 : 중심점 좌표, 튜플
        

> **[예제 7-4] 컨투어를 감싸는 도형 그리기(7.4_cntr_bound_fit.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> # Read image and convert to grayscale, then convert to binary scale
> img = cv2.imread('./img/lightning.png')
> img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
> ret, th = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
> 
> # Find contour
> contours, hr = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
> 
> contour = contours[0]
> 
> # Display rectangle surrounding contour(black)
> x, y, w, h = cv2.boundingRect(contour)
> cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 3)
> 
> # Display minimum enclosing rectangle(green)
> rect = cv2.minAreaRect(contour)
> box = cv2.boxPoints(rect)    # Convert center point and angle into coordinates of four vertices
> box = np.int0(box)           # Convert to Integer
> cv2.drawContours(img, [box], -1, (0, 255, 0), 3)
> 
> # Display minimum enclosing circle(blue)
> (x, y), radius = cv2.minEnclosingCircle(contour)
> cv2.circle(img, (int(x), int(y)), int(radius), (255, 0, 0), 2)
> 
> # Display minimum enclosing triangle(pink)
> ret, tri = cv2.minEnclosingTriangle(np.float32(contour))
> cv2.polylines(img, [np.int32(tri)], True, (255, 0, 255), 2)
> 
> # Display minimum enclosing ellipse(yellow)
> ellipse = cv2.fitEllipse(contour)
> cv2.ellipse(img, ellipse, (0, 255, 255), 3)
> 
> # Display a straight line passing through the center point(red)
> [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
> cols, rows = img.shape[:2]
> cv2.line(img, (0, int(0-x*(vy/vx) + y)), (cols-1, int((cols-x)*(vy/vx) + y)), (0, 0, 255), 2)
> 
> # Output result
> cv2.imshow('Bound Fit shapes', img)
> cv2.waitKey()
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled 3](https://user-images.githubusercontent.com/69300448/226506481-052f7f65-6d4d-4675-b7c3-5b0e1c7e17dc.png)
> 

### 7.1.2 컨투어 단순화

실생활에서 얻은 영상은 물체가 정확히 표현되는 경우보다 노이즈와 침식이 일어나는 경우가 더 많다. 그래서 컨투어도 정확한 컨투어보다는 부정확하게 단순화한 컨투어가 쓸모있는 경우가 더 많다.

OpenCV는 오차범위 내 근사값으로 컨투어를 계산해주는 함수를 제공한다.

- **approx = cv2.approxPolyDP(contour, epsilon, closed)**
    - contour : 대상 컨투어 좌표
    - epsilon : 근사값 정확도, 오차 범위
    - closed : 컨투어 닫힘 여부
    - approx : 근사 계산한 컨투어 좌표

> **[예제 7-5] 근사 컨투어(7.5_cntr_approximate.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> img = cv2.imread('./img/bad_rect.png')
> img2 = img.copy()
> 
> # Convert to gray scale and binary scale
> img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
> ret, th = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
> 
> # Find contour
> contours , hr = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
> contour = contours[0]
> 
> # Designation the error range as 0.05 of the total circumference
> epsilon = 0.05 * cv2.arcLength(contour, True)
> 
> # Calculate an approximate contour
> approx = cv2.approxPolyDP(contour, epsilon, True)
> 
> # Draw contour line
> cv2.drawContours(img, [contour], -1, (0, 255, 0), 3)
> cv2.drawContours(img2, [approx], -1, (0, 255, 0), 3)
> 
> # Output result
> cv2.imshow('contour', img)
> cv2.imshow('approx', img2)
> cv2.waitKey()
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled 4](https://user-images.githubusercontent.com/69300448/226506505-015800af-d875-493f-9acf-882faf027aca.png)
> 

컨투어를 단순화하는 다른 방법은 **볼록 선체(convex hull)** 를 만드는 것이다. 볼록 선체는 어느 한 부분도 오목하지 않은 상태를 말하는 것으로 대상 객체를 완전히 포함하므로 객체의 외곽 영역을 찾는 데 좋다.

- **hull = cv2.convexHull(points[, hull, clockwise, returnPoints])** : 블록 선체 찾기
    - points : 입력 컨투어
    - hull : 볼록 선체 결과
    - clockwise : 방향 지정(True : 시계 방향)
    - returnPoints : 결과 좌표 형식 선택
        - True : 볼록 선체 좌표 반환
        - False : 입력 컨투어 중에 볼록 선체에 해당하는 인덱스 반환
- **retval = cv2.isContourConvex(contour)** : 볼록 선체 만족 여부 확인
    - retval : True인 경우 볼록 선체 만족
- **defects = cv2.convexityDefects(contour, convexhull)** : 볼록 선체 결함 찾기
    - contour : 입력 컨투어
    - convexhull : 볼록 선체에 해당하는 컨투어의 인덱스
    - defects : 볼록 선체 결함이 있는 컨투어의 배열 인덱스, N x 1 x 4 배열
    - [start, end, farthest, distance]
        - start : 오목한 각이 시작되는 컨투어의 인덱스
        - end : 오목한 각이 끝나는 컨투어의 인덱스
        - farthest : 볼록 선체에서 가장 먼 오목한 지점의 컨투어 인덱스
        - distance : farthest와 볼록 선체와의 거리, 8비트 고정 소수점(distance/256.0)

> **[예제 7-6] 볼록 선체(7.6_cntr_convexhull.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> img = cv2.imread('./img/hand.jpg')
> img2 = img.copy()
> 
> # Convert to gray scale and binary scale
> gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
> ret, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
> 
> # Find and draw a contour
> contours, hr = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
> cntr = contours[0]
> cv2.drawContours(img, [cntr], -1, (0, 255, 0), 1)
> 
> # Find and draw a convex hull(based on coordinates)
> hull = cv2.convexHull(cntr)
> cv2.drawContours(img2, [hull], -1, (0, 255, 0), 1)
> 
> # Check satisfaction with the convex hull
> print(cv2.isContourConvex(cntr), cv2.isContourConvex(hull))
> 
> >>> False True
> 
> # Find a convex hull(based on index)
> hull2 = cv2.convexHull(cntr, returnPoints=False)
> 
> # Find a convex hull defects
> defects = cv2.convexityDefects(cntr, hull2)
> 
> # Iterate convex hull defects
> for i in range(defects.shape[0]):
>     # start, end, farthest point, distance
>     startP, endP, farthestP, distance = defects[i, 0]
>     
>     # Get the coordinates of the farthest point
>     farthest = tuple(cntr[farthestP][0])
>     
>     # Convert distance to floating point number
>     dist = distance/256.0
>     
>     # When the distance is greater than 1
>     if dist > 1 :
>         # Mark red dot
>         cv2.circle(img2, farthest, 3, (0, 0, 255), -1)
> 
> # Display result image
> cv2.imshow('contour', img)
> cv2.imshow('convex hull', img2)
> cv2.waitKey()
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled 5](https://user-images.githubusercontent.com/69300448/226506548-5fc19a7b-e737-47a0-bbfb-3a80e5ca28bc.png)
> 

### 7.1.3 컨투어와 도형 매칭

서로 다른 물체의 컨투어를 비교하면 두 물체가 얼마나 비슷한 모양인지를 알 수 있다. 이를 위해서는 위치, 크기, 그리고 방향에 불변하는 휴 모멘트들을 이용한 복잡한 연산이 필요하다.

OpenCV에서 제공하는 함수를 이용하면 간단히 할 수 있다.

- retval = cv2.matchShapes(contour1, contour2, method, parameter) : 두 개의 컨투어로 도형 매칭
    - contour1, contour2 : 비교할 두 개의 컨투어
    - method : 휴 모멘트 비교 알고리즘 선택 플래그
        - cv2.CONTOURS_MATCH_I1
        - cv2.CONTOURS_MATCH_I2
        - cv2.CONTOURS_MATCH_I3
    - parameter : 알고리즘에 전달을 위한 예비 인수, 현재 지원 안됨(0으로 고정)
    - retval : 닮음 정도, 0 = 동일, 클수록 다름
    

> **[예제 7-7] 도형 매칭으로 비슷한 도형 찾기(7.7_cntr_matchShape.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> # Read image for matching
> target = cv2.imread('./img/4star.jpg')
> shapes = cv2.imread('./img/shapestomatch.jpg')
> 
> # Convert to gray scale
> target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
> shapes_gray = cv2.cvtColor(shapes, cv2.COLOR_BGR2GRAY)
> 
> # Convert to binary scale
> ret, target_th = cv2.threshold(target_gray, 127, 255, cv2.THRESH_BINARY_INV)
> ret, shapes_th = cv2.threshold(shapes_gray, 127, 255, cv2.THRESH_BINARY_INV)
> 
> # Find contours
> cntr_target, _ = cv2.findContours(target_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
> cntr_shapes, _ = cv2.findContours(shapes_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
> 
> # Loop for matching each shape
> matchs = []    # List for saving contour and matching score
> for cntr in cntr_shapes:
>     # Match the target shape with one of the several shapes
>     match = cv2.matchShapes(cntr_target[0], cntr, cv2.CONTOURS_MATCH_I2, 0.0)
>     # Save matching scores and contours of the corresponding shapes as a pair
>     matchs.append((match, cntr))
>     # Display matching score at the starting point of the contour of the corresponding shape
>     cv2.putText(shapes, f'{match:.2f}', tuple(cntr[0][0]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
> 
> # Sort by matching score
> matchs.sort(key=lambda x : x[0])
> 
> # Draw a line on the contour of the shape with the lowest matching score
> cv2.drawContours(shapes, [matchs[0][1]], -1, (0, 255, 0), 3)
> cv2.imshow('target', target)
> cv2.imshow('Match Shape', shapes)
> cv2.waitKey()
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled 6](https://user-images.githubusercontent.com/69300448/226506583-7f05eb5f-f8cc-484e-b1f0-238a7a5d655a.png)
>

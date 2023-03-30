# 7.3 연속 영역 분할

외곽 경계를 이용해서 객체 영역을 분할하는 방법은 실생활에서 경계선이 분명하지 않아 문제를 해결하기 어려운 경우가 많다. 그래서 영상 분할에서는 연속된 영역을 찾아 분할하는 방법도 함께 사용한다.

### 7.3.1 거리 변환

영상에서 물체의 영역을 정확히 파악하기 위한 방법으로 물체의 최중심점을 찾는 것이 중요하다. 사람이나 동물의 뼈대 같은 것으로 흔히 **스켈레톤(skeleton)** 이라고 한다. 스켈레톤을 검출하는 방법 중 하나가 주변 경계로부터 가장 멀리 떨어진 곳을 찾는 **거리 변환** 이다.

**거리 변환(distance transform)** 은 바이너리 스케일 이미지에서 픽셀값이 0인 위치에 0으로 시작해서 멀어질 때마다 1씩 증가하는 방식으로 경계로부터 가장 먼 픽셀이 가장 큰 값을 갖게 하는 변환이다.

- cv2.distanceTransform(src, distanceType, maskSize)
    - src : 입력 영상, 바이너리 스케일
    - distanceType : 거리 계산 방식 선택
        - cv2.DIST_L2, cv2.DIST_L1, cv2.DIST_L12, cv2.DIST_FAIR, cv2.DIST_WELSCH, cv2.DIST_HUBER
    - maskSize : 거리 변환 커널 크기

> **[예제 7-11] 거리 변환으로 전신 스켈레톤 찾기(7.11_distanceTrans.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> # Read a image and convert to binary scale
> img = cv2.imread('./img/full_body.jpg', cv2.IMREAD_GRAYSCALE)
> _, biimg = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
> 
> # Distance transform
> dst = cv2.distanceTransform(biimg, cv2.DIST_L2, 5)
> 
> # Normalize distance values to a range of 0 to 255
> dst = (dst/(dst.max()-dst.min()) * 255).astype(np.uint8)
> 
> # Find complete skeleton using threshold on distance values
> skeleton = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, -3)
> 
> # Output result
> merged = np.hstack((img, dst, skeleton))
> cv2.imshow('origin, dist, skel', merged)
> cv2.waitKey(0)
> cv2.destroyAllWindows()
> ```
> ![Untitled](https://user-images.githubusercontent.com/69300448/228851606-a2ecb2e0-ae7d-438b-bec8-2d3339910603.png)
> 

### 7.3.2 연결 요소 레이블링

연결된 요소들끼리 분리하는 방법으로 **레이블링(labeling)** 이라는 방법이 있다. 바이너리 스케일 이미지에서 픽셀값이 0으로 끊어지지 않는 영역끼리 같은 값을 부여해서 분리하는 방법이다.

- **retval, labels = cv2.connectedComponents(src[, labels, connectivity=8, ltype])** : 연결 요소 레이블링과 개수 반환
    - src : 입력 영상, 바이너리 스케일 이미지
    - labels : 레이블링 된 입력 영상과 같은 크기의 배열
    - connectivity : 연결성을 검사할 방향 개수(4, 8 중 선택)
    - ltype : 결과 레이블 배열 dtype
    - retval : 레이블 개수
- **retval, labels, stats, centroids = cv2.connectedComponentsWithStats(src[, labels, stats, centroids, connectivity, ltype])** : 레이블링과 각종 상태 정보 반환
    - stats : N x 5 행렬(N : 레이블 개수)
        - [x 좌표, y 좌표, 폭, 높이, 넓이]
    - centroids : 각 레이블의 중심점 좌표, N x 2 행렬(N : 레이블 개수)

> **[예제 7-12] 연결된 영역 레이블링(7.12_connected_label.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> # Read a image
> img = cv2.imread('./img/shapes_donut.png')
> # Create a result image
> img2 = np.zeros_like(img)
> # Convert to gray and binary scale
> gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
> _, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
> 
> # Apply connected component labeling
> cnt, labels = cv2.connectedComponents(th)
> # retval, labels, stats, cent = cv2.connectedComponentsWithStats(th)
> 
> # Iterate as many as the number of labels
> for i in range(cnt):
>     # Apply random colors to areas with the same label
>     img2[labels==i] = [int(j) for j in np.random.randint(0, 255, 3)]
> 
> # Output result
> merged = np.hstack((img, img2))
> cv2.imshow('origin, labeled', merged)
> cv2.waitKey()
> cv2.destroyAllWindows()
> ```
> ![Untitled 1](https://user-images.githubusercontent.com/69300448/228851674-9b402256-985f-497f-a884-d34983741fec.png)
> 

### 7.3.3 색 채우기

OpenCV는 연속되는 영역에 같은 색상을 채워 넣을 수 있는 함수를 제공한다.

- **retval, img, mask, rect = cv2.floodFill(img, mask, seed, newVal[, loDiff, upDiff, flags])**
    - img : 입력 영상, 1 또는 3채널
    - mask : 입력 영상보다 2 x 2 픽셀이 더 큰 배열, 0이 아닌 영역을 만나면 채우기 중지
    - seed : 채우기 시작할 좌표
    - newVal : 채우기에 사용할 색상값
    - loDiff, upDiff : 채우기 진행을 결정할 최소/최대 차이 값
    - flags : 채우기 방식 선택 플래그
        - 4 또는 8 방향 채우기
        - cv2.FLOODFILL_MASK_ONLY : img가 아닌 mask에만 채우기 적용
            - 채우기에 사용할 값을 8~16 비트에 포함시켜야 함
        - cv2.FLOODFILL_FIXED_RANGE : 이웃 픽셀이 아닌 seed 픽셀과 비교
    - retval : 채우기 한 픽셀의 개수
    - rect : 채우기가 이뤄진 영역을 감싸는 사각형
1. 위 함수는 img 영상의 seed 좌표에서부터 시작해서 newVal의 값으로 채우기 시작한다.
2. 이웃하는 픽셀에 채우기를 계속하려면 현재 픽셀이 이웃 픽셀의 loDiff를 뺀 값보다 크거나 같고, upDiff를 더한 값보다 작거나 같아야 한다. 이 값을 생략하면 seed와 같은 값을 갖는 이웃 픽셀만 채우기를 진행한다.

$$
src(x', y')-loDiff \leq src(x, y) \leq src(x', y') + upDiff
$$

- $src(x, y)$ : 현재 픽셀
- $src(x', y')$ : 이웃 픽셀
1. flags에 cv2.FLOODFILL_FIXED_RAGE가 포함되면 이웃한 픽셀이 아니라 seed 픽셀과 비교한다.
flags에 cv2.FLOODFILL_MASK_ONLY가 포함되어 있으면 img에 채우기를 하지 않고 mask에만 채우기를 한다.
2. mask는 입력 영상보다 가로/세로 방향으로 2픽셀씩 더 커야하고 0이 아닌 값을 가진 영역을 만나면 채우기 조건을 만족하더라도 더 이상 채우기를 진행하지 않는다. 따라서 경계를 검출한 바이너리 스케일 이미지를 mask로 사용하면 경계를 넘지 않게 할 수 있다.

```python
flags = 8 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8)
```

> **[예제 7-13] 마우스로 색 채우기(7.13_flood_fill.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> img = cv2.imread('./img/taekwonv1.jpg')
> rows, cols = img.shape[:2]
> 
> # Create mask, 2 pixels larger than the original image
> mask = np.zeros((rows+2, cols+2), np.uint8)
> 
> # Colors to use for filling
> newVal = (255, 255, 255)
> 
> # Minimum/maximum difference value
> loDiff, upDiff = (10, 10, 10), (10, 10, 10)
> 
> # Mouse event handling function
> def onMouse(event, x, y, flags, param):
>     global mask, img
>     if event == cv2.EVENT_LBUTTONDOWN:
>         seed = (x, y)
>         # Apply fill color
>         retval = cv2.floodFill(img, mask, seed, newVal, loDiff, upDiff)
>         # Display change results for fillable fields
>         cv2.imshow('img', img)
> 
> # Output result
> cv2.imshow('img', img)
> cv2.setMouseCallback('img', onMouse)
> cv2.waitKey()
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled 2](https://user-images.githubusercontent.com/69300448/228851729-9be5a7a1-8fe1-419e-a959-28ddbc6ee0f0.png)
> 

### 7.3.4 워터셰드

**워터셰드(watershed)** 는 우리말로는 분수령 혹은 분수계라고 한다. 강물이 한 줄기로 흐르다가 갈라지는 경계를 의미한다. 

영상 처리에서 워터셰드는 경계를 찾는 방법 중 하나로 픽셀값의 크기를 산과 골짜기 같은 높고 낮은 지형으로 보고 물을 채워서 그 물이 만나는 곳을 경계로 찾는 방식이다.

색 채우기와 비슷한 방식으로 연속된 영역을 찾는 것이다. 이때 처음 찾을 지점인 seed를 하나가 아닌 여러 곳을 사용하고 이것을 **마커**라고 한다.

- **markers = cv2.watershed(img, markers)**
    - img : 입력 영상
    - markers : 마커, 입력 영상과 크기가 같은 1차원 배열(int32)

markers의 값은 경계를 찾고자 하는 픽셀 영역은 0을 갖게 하고 연결된 영역이 확실한 픽셀에 대해서는 동일한 양의 정수를 값으로 갖게 한다. watershed 함수는 markers에 0이 아닌 값들을 이용해서 같은 영역 모두를 같은 값으로 채우고 그 경계는 -1로 채워서 반환한다. 반환된 마커를 이용해서 원본 영상에 연결된 나머지 영역과 경계를 찾을 수 있다.

> **[예제 7-14] 마우스와 워터셰드로 배경 분리(7.14_watershed.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> img=cv2.imread('./img/taekwonv1.jpg')
> rows,cols = img.shape[:2]
> img_draw=img.copy()
> 
> # Create marker, Initialize all elements to 0
> marker = np.zeros((rows, cols), np.int32)
> markerId = 1
> colors = []
> isDragging = False    # Variable to check if dragging is happening
> 
> # Mouse event handling function
> def onMouse(event, x, y, flags, param):
>     global img_draw, marker, markerId, isDragging, colors
>     if event == cv2.EVENT_LBUTTONDOWN:
>         isDragging = True
>         # Save ID of each marker and color value of its current position as pairs
>         colors.append((markerId, img[y, x]))
>     elif event == cv2.EVENT_MOUSEMOVE:
>         if isDragging:
>             # Fill in the marker coordinates corresponding to the mouse coordinates with the same marker ID
>             marker[y, x] = markerId
>             # Print the marked locations as red dots
>             cv2.circle(img_draw, (x, y), 3, (0, 0, 255), -1)
>             cv2.imshow('watershed', img_draw)
>     elif event == cv2.EVENT_LBUTTONUP:
>         if isDragging:
>             isDragging = False
>             # Increase marker ID for selecting next marker
>             markerId += 1
>     elif event == cv2.EVENT_RBUTTONDOWN:
>         # Apply watershed using the collected marker
>         cv2.watershed(img, marker)
>         # Display the boundaries marked with -1 in green color
>         img_draw[marker == -1] = (0, 255, 0)
>         for mid, color in colors:    # Iterate for the numbers of selected marker IDs
>             # Fill the area with the same marker ID value with selected color of the marker
>             img_draw[marker==mid] = color
>         cv2.imshow('watershed', img_draw)
> 
> # Output result
> cv2.imshow('watershed', img)
> cv2.setMouseCallback('watershed', onMouse)
> cv2.waitKey()
> cv2.destroyAllWindows()
> ```
> ![Untitled 3](https://user-images.githubusercontent.com/69300448/228851777-7957bc10-c92a-434b-a910-f4451f0ab564.png)
> 

워터셰드는 경계 검출이 어려운 경우 배경으로 확신할 수 있는 픽셀과 전경으로 확신할 수 있는 픽셀로 경계를 찾을 수 있다.

### 7.3.5 그랩컷

**그랩컷(grabcut)** 은 그래프 컷(graph cut)을 기반으로 하는 알고리즘을 확장한 것으로, 사용자가 전경으로 분리할 대상 객체가 있는 사각형 좌표를 주면 대상 객체와 배경의 색상 분포를 추정해서 동일한 레이블을 가진 연결된 영역에서 배경과 전경을 분리한다.

- **mask, bgdModel, fgdModel = cv2.grabCut(img, mask, rect, bgdModel, fgdModel, iterCount[, mode])**
    - img : 입력 영상
    - mask : 입력 영상과 크기가 같은 1채널 배열, 배경과 전경을 구분하는 값 저장
        - cv2.GC_BGD : 확실한 배경(0)
        - cv2.GC_FGD : 확실한 전경(1)
        - cv2.GC_PR_BGD : 아마도 배경(2)
        - cv2.GC_PR_FGD : 아마도 전경(3)
    - rect : 전경이 있을 것으로 추측되는 영역의 사각형 좌표, 튜플(x1, y1, x2, y2)
    - bgdModel, fgdModel : 함수 내에서 사용할 임시 배열 버퍼(재사용할 경우 수정하지 말 것)
    - iterCount : 반복 횟수
    - mode : 동작 방법
        - cv2.GC_INIT_WITH_RECT : rect에 지정한 좌표를 기준으로 그랩컷 수행
        - cv2.GC_INIT_WITH_MASK : mask에 지정한 값을 기준으로 그랩컷 수행
        - cv2.GC_EVAL : 재시도

> **[예제 7-15] 마우스와 그랩컷으로 배경 분리(7.15_grabcut.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> img = cv2.imread('./img/taekwonv1.jpg')
> img_draw = img.copy()
> # Create mask
> mask = np.zeros(img.shape[:2], dtype=np.uint8)
> # Initialize coordinates of a rectangular area
> rect = [0, 0, 0, 0]
> # Grabcut initial mode
> mode = cv2.GC_EVAL
> 
> # Background and foreground model buffer
> bgdmodel = np.zeros((1, 65), np.float64)
> fgdmodel = np.zeros((1, 65), np.float64)
> 
> # Mouse event handling function
> def onMouse(event, x, y, flags, param):
>     global mouse_mode, rect, mask, mode
>     if event == cv2.EVENT_LBUTTONDOWN:
>         if flags <= 1:                      # If do not press any key
>             mode = cv2.GC_INIT_WITH_RECT    # Starting drag, rectangle mode
>             rect[:2] = x, y                 # Save starting coordinates
>     elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
>         if mode == cv2.GC_INIT_WITH_RECT:   # Dragging in progress
>             img_temp = img.copy()
>             # Display a drag rectangle
>             cv2.rectangle(img_temp, (rect[0], rect[1]), (x, y), (0, 255, 0), 2)
>             cv2.imshow('img', img_temp)
>         elif flags > 1:                             # Pressed state of a key
>             mode = cv2.GC_INIT_WITH_MASK            # mask mode
>             if flags & cv2.EVENT_FLAG_CTRLKEY:      # Ctrl key, definitely foreground
>                 # Display white dots on the screen
>                 cv2.circle(img_draw, (x, y), 3, (255, 255, 255), -1)
>                 # Fill the mask with GC_FGD
>                 cv2.circle(mask, (x, y), 3, cv2.GC_FGD, -1)
>             if flags & cv2.EVENT_FLAG_SHIFTKEY:     # Shift key, definitely background
>                 # Display black dots on the screen
>                 cv2.circle(img_draw, (x, y), 3, (0, 0, 0), -1)
>                 # Fill the mask with GC_BGD
>                 cv2.circle(mask, (x, y), 3, cv2.GC_BGD, -1)
>             cv2.imshow('img', img_draw)    # Display drawn image
>     elif event == cv2.EVENT_LBUTTONUP:
>         if mode == cv2.GC_INIT_WITH_RECT:
>             rect[2:] = x, y            # Collect the last coordinate of a rectangle
>             # Draw rectangle and dispay it
>             cv2.rectangle(img_draw, (rect[0], rect[1]), (x, y), (255, 0, 0), 2)
>             cv2.imshow('img', img_draw)
>         # Apply grabcut
>         cv2.grabCut(img, mask, tuple(rect), bgdmodel, fgdmodel, 1, mode)
>         img2 = img.copy()
>         # Fill the area marked as 'definitely background' and 'probably background' with 0 on the mask
>         img2[(mask==cv2.GC_BGD) | (mask==cv2.GC_PR_BGD)] = 0
>         cv2.imshow('grabcut', img2)    # Print final result
>         mode = cv2.GC_EVAL             # Reset grabcut mode
> 
> # Display initial screen and register mouse events
> cv2.imshow('img', img)
> cv2.setMouseCallback('img', onMouse)
> while True:
>     if cv2.waitKey() & 0xFF == 27:
>         break
> cv2.destroyAllWindows()
> ```
> ![Untitled 4](https://user-images.githubusercontent.com/69300448/228851831-d6d1cc0d-bc25-44de-b6f2-2b246ef988bf.png)
> 

### 7.3.6 평균 이동 필터

영상의 일정한 반경 크기의 커널로 픽셀의 평균값을 커널의 중심으로 바꿔서 이동하는 것을 반복하다보면 그 주변에서 가장 밀집한 곳을 찾을 수 있다. 특정 공간 내의 분포의 피크(peak)를 찾는 방법을 **평균 이동** 이라고 한다. 

이동을 시작한 지점에서 중지한 지점까지를 하나로 묶으면 연결된 영역을 찾을 수 있다. 같은 방법으로 가장 빈도가 많은 색상을 구해서 연결된 영역의 모든 픽셀값으로 바꾸면 연결된 영역을 구분할 수 있다.

OpenCV에서는 함수로 이 기능을 제공한다. 이 함수는 내부적으로 이미지 피라미드를 만들어 작은 영상의 평균 이동 결과를 큰 영상에 적용할 수 있어서 이름 앞에 pyr이 붙는다.

- dst = cv2.pyrMeanShiftFiltering(src, sp, sr[, dst, maxLevel, termcrit])
    - src : 입력 영상(컬러와 그레이 스케일 모두 가능)
    - sp : 공간 윈도 반지름 크기
    - sr : 색상 윈도 반지름 크기(너무 작으면 원본과 별 차이가 없고, 너무 크면 영역이 무너지는 결과)
    - maxLevel : 이미지 피라미드 최대 레벨
    - termcrit : 반복 중지 요건
        - type=cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS : 중지 형식
            - cv2.TERM_CRITERIA_EPS : 정확도가 최소 정확도(epsilon)보다 작아지면 중지
            - cv2.TERM_CRITERIA_MAX_ITER : 최대 반복 횟수(max_iter)에 도달하면 중지
            - cv2.TERM_CRITERIA_COUNT : cv2.TERM_CRITERIA_MAX_ITER와 동일
        - max_iter=5 : 최대 반복 횟수
        - epsilon=1 : 최소 정확도

> **[예제 7-16] 평균 이동 세그멘테이션 필터(7.16_mean_shift.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> img = cv2.imread('./img/taekwonv1.jpg')
> 
> # Trackbar event handling function
> def onChange(x):
>     # Collect the selected values for sp, sr, level
>     sp = cv2.getTrackbarPos('sp', 'img')
>     sr = cv2.getTrackbarPos('sr', 'img')
>     lv = cv2.getTrackbarPos('lv', 'img')
>     
>     # Apply MeanShiftFiltering
>     mean = cv2.pyrMeanShiftFiltering(img, sp, sr, None, lv)
>     # Print converted image
>     cv2.imshow('img', np.hstack((img, mean)))
> 
> # Output initial image
> cv2.imshow('img', np.hstack((img, img)))
> # Connect trackbar handle function
> cv2.createTrackbar('sp', 'img', 0, 100, onChange)
> cv2.createTrackbar('sr', 'img', 0, 100, onChange)
> cv2.createTrackbar('lv', 'img', 0, 5, onChange)
> cv2.waitKey()
> cv2.destroyAllWindows()
> ```
> ![Untitled 5](https://user-images.githubusercontent.com/69300448/228851869-c15d529f-4e9e-45e4-8956-33ed9dba5997.png)
>

# 4.1 관심영역(ROI)

이미지에 어떤 연산을 적용해서 새로운 이미지나 정보를 얻어내려고 할 때, 전체 이미지를 대상으로 연산을 하는 것보다 관심이 있는 부분만 잘라내서 하는 것이 훨씬 효과적이다.

관심있는 영역만 잘라내서 연산을 하면,

- 연산할 데이터의 양을 줄이고 수행 시간을 단축시키는 이점
- 데이터의 양이 줄어 들면 그 형태도 단순해지므로 적용해야 하는 알고리즘도 단순해지는 이점
- 이미지 연산은 항상 좌표를 기반으로 해야 하는데, 그 영역이 클 때보다 작을 때 좌표 구하기가 쉽다는 이점

### 4.1.1 관심영역 지정

전체 이미지에서 연산과 분석의 대상이 되는 영역을 관심영역(Region Of Interest, ROI) 라고 한다.

전체 이미지가 img라는 변수에 있을 때, 관심 있는 영역의 좌표가 x, y이고 영역의 폭이 w, 높이가 h라면 관심영역을 지정하는 코드는 아래와 같다.

```python
roi = img[y:y+h, x:x+w]
```

**Numpy를 이용해서 관심영역을 지정할 때 주의해야 할 사항 두가지**

- Numpy 배열은 행(row), 열(column) 순으로 접근하므로, 이미지는 높이(height), 폭(width) 순으로 지정해야 한다.
- Numpy 배열의 슬라이싱은 원본의 참조를 반환하기 때문에 슬라이싱 연산해서 얻은 결과의 값을 수정하면 원본 배열 객체에서도 값이 달라진다. 원본과 무관한 새로운 작업을 하려면 반드시 copy() 함수로 복제본을 생성해서 작업해야한다.

> **[예제 4-1] 관심영역 지정(4.1_roi.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> import matplotlib.pyplot as plt
> 
> img = cv2.imread('./img/sunset.jpg')
> plt.imshow(img[:, :, ::-1])
> plt.xticks([]), plt.yticks([])
> plt.show()
> 
> # roi 좌표
> x = 320
> y = 150
> w = 50
> h = 50
> roi = img[y:y+h, x:x+w]
> 
> print(roi.shape)
> 
> # roi에 사각형 그리기
> cv2.rectangle(roi, (0, 0), (h-1, w-1), (0, 255, 0))
> 
> cv2.imshow('img', img)
> 
> cv2.waitKey(0)
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled](./Image/[OpenCV] 관심영역(ROI)/Untitled.png)
> 

> **[예제 4-2] 관심영역 복제 및 새 창 띄우기(4.2_roicopy.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> import matplotlib.pyplot as plt
> 
> img = cv2.imread('./img/sunset.jpg')
> plt.imshow(img[:, :, ::-1])
> plt.xticks([]), plt.yticks([])
> plt.show()
> 
> # roi 좌표
> x = 320
> y = 150
> w = 50
> h = 50
> roi = img[y:y+h, x:x+w]  # roi 지정
> img2 = roi.copy()        # roi 복제
> 
> # 새로운 좌표에 roi 추가하기(태양 2개 만들기)
> img[y:y+h, x+w:x+w+w] = roi
> 
> # 2개의 태양 영역에 사각형 표시
> cv2.rectangle(img, (x, y), (x+w+w, y+h), (0, 255, 0))
> 
> # 이미지 출력
> cv2.imshow('img', img)
> cv2.imshow('roi', img2)
> 
> cv2.waitKey(0)
> cv2.destroyAllWindows()
> 
> # matplotlib으로 출력
> plt.imshow(img[:,:,::-1])
> plt.xticks([]), plt.yticks([])
> plt.show()
> 
> fig = plt.figure(figsize=(3, 3))
> plt.imshow(img2[:,:,::-1])
> plt.xticks([]), plt.yticks([])
> plt.show()
> ```
> 
> ![Untitled](4%201%20%E1%84%80%E1%85%AA%E1%86%AB%E1%84%89%E1%85%B5%E1%86%B7%E1%84%8B%E1%85%A7%E1%86%BC%E1%84%8B%E1%85%A7%E1%86%A8(ROI)%2060ac859f536b4d5993e532f2c562bb9b/Untitled%201.png)
> 

### 4.1.2 마우스로 관심영역 지정

마우스를 이용해 원하는 영역을 직접 지정하고 좌표를 알아내는 방법을 알아보자.

> **[예제 4-3] 마우스로 관심영역 지정(4.3_roi_crop_mouse.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> isDragging = False
> x0, y0, w, h = -1, -1, -1, -1
> blue, red = (255, 0, 0), (0, 0, 255)
> 
> # 마우스 이벤트 핸들 함수
> def onMouse(event, x, y, flags, param):
>     global isDragging, x0, y0, img
>     # 드래그 시작
>     if event == cv2.EVENT_LBUTTONDOWN:
>         isDragging = True
>         x0 = x
>         y0 = y
>     # 마우스 드래그
>     elif event == cv2.EVENT_MOUSEMOVE:
>         # 드래그 진행 중
>         if isDragging:
>             img_draw = img.copy()        # 사각형 그림 표현을 위한 이미지 복제
>             cv2.rectangle(img_draw, (x0, y0), (x, y), blue, 2)  # 드래그 진행 영역 표시
>             cv2.imshow('img', img_draw)  # 사각형으로 표시된 그림 화면 출력
>     # 드래그 종료
>     elif event == cv2.EVENT_LBUTTONUP:
>         if isDragging:
>             isDragging = False
>             w = x - x0             # 너비 계산
>             h = y - y0             # 높이 계산
>             print(f'x:{x0}, y:{y0}, w:{w}, h:{h}')
> 
>             if w > 0 and h > 0:
>                 img_draw = img.copy()
>                 cv2.rectangle(img_draw, (x0, y0), (x, y), red, 2)
>                 cv2.imshow('img', img_draw)
>                 roi = img[y0:y0+h, x0:x0+w]
>                 cv2.imshow('cropped', roi)
>                 cv2.moveWindow('cropped', 0, 0)
>                 cv2.imwrite('./result/cropped.jpg', roi)
>                 print('cropped.')
>             # 드래그 방향이 잘못된 경우
>             else:
>                 cv2.imshow('img', img)
>                 print('좌측 상단에서 우측 하단으로 영역을 드래그하세요.')
> 
> img = cv2.imread('./img/sunset.jpg')
> 
> cv2.imshow('img', img)
> cv2.setMouseCallback('img', onMouse)
> cv2.waitKey()
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled](4%201%20%E1%84%80%E1%85%AA%E1%86%AB%E1%84%89%E1%85%B5%E1%86%B7%E1%84%8B%E1%85%A7%E1%86%BC%E1%84%8B%E1%85%A7%E1%86%A8(ROI)%2060ac859f536b4d5993e532f2c562bb9b/Untitled%202.png)
> 

OpenCV 3는 관심영역을 지정하기 위한 새로운 함수를 제공한다.

마우스로 영역을 선택하고, 키보드의 스페이스 또는 엔터 키를 누르면 선택한 영역의 x, y 좌표와 영역의 폭과 높이를 튜플에 담아 반환한다. 선택을 취소하고 싶으면 키보드의 ‘c’ 키를 누르면 된다.

- ret = cv2.selectROI([win_name,] img[, snowCrossHair=True, fromCenter=False])
    - win_name : ROI 선택을 진행할 창의 이름, str
    - img : ROI 선택을 진행할 이미지, Numpy ndarray
    - showCrossHair : 선택 영역 중심에 십자 모양 표시 여부
    - fromCenter : 마우스 시작 지점을 영역의 중심으로 지정
    - ret : 선택한 영역 좌표와 크기(x, y, w, h), 선택을 취소한 경우 모두 0
    

> **[예제 4-4] selectROI로 관심영역 지정(4.4_roi_select_img.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> img = cv2.imread('./img/sunset.jpg')
> 
> # cv2.selectROI - 엔터, 스페이스 : 저장, c : 취소
> x, y, w, h = cv2.selectROI('img', img, False)
> print(w, h)
> if w and h:
>     roi = img[y:y+h, x:x+w]
>     cv2.imshow('cropped', roi)                # roi 지정 영역을 새 창으로 표시
>     cv2.moveWindow('cropped', 0, 0)           # 새 창을 좌측 상단으로 이동
>     cv2.imwrite('./result/cropped2.jpg', roi) # ROI 영역만 파일로 저장
>     
> cv2.imshow('img', img)
> cv2.waitKey(0)
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled](4%201%20%E1%84%80%E1%85%AA%E1%86%AB%E1%84%89%E1%85%B5%E1%86%B7%E1%84%8B%E1%85%A7%E1%86%BC%E1%84%8B%E1%85%A7%E1%86%A8(ROI)%2060ac859f536b4d5993e532f2c562bb9b/Untitled%203.png)
>
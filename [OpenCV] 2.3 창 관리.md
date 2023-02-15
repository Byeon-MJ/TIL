# 2.3 창 관리

OpenCV가 제공하는 창 관리 API들, 창을 열 때 사용한 이름을 기반으로 연결되는 것이 특징이다.

- **cv2.namedWindow(title [, option])** : 이름을 갖는 창 열기
    - title : 창 이름, 제목 줄에 표시
    - option : 창 옵션, ‘cv2.WINDOW_’로 시작
        - cv2.WINDOW_NORMAL : 임의의 크기, 사용자 창 크기 조정 기능
        - cv2.WINDOW_AUTOSIZE : 이미지와 같은 크기, 창 크기 재조정 불가능
- **cv2.moveWindow(title, x, y)** : 창 위치 이동
    - title : 위치를 변경할 창의 이름
    - x, y : 이동할 창의 위치
- **cv2.resizeWindow(title, width, height)** : 창 크기 변경
    - title : 크기를 변경할 창의 이름
    - width, height : 크기를 변경할 창의 폭과 높이
- **cv2.destroyWindow(title)** : 창 닫기
    - title : 닫을 대상 창 이름
- **cv2.destroyAllWindows()** : 열린 모든 창 닫기

> **[예제 2-15] 창 관리 API 사용하기(2.15_win.ipynb)**
> 
> 
> ```python
> import cv2
> 
> file_path = './img/girl.jpg'
> 
> # 이미지를 기본 값으로 읽기
> img = cv2.imread(file_path)
> # 이미지를 그레이 스케일로 읽기
> img_gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
> 
> # origin 이름으로 창 생성
> cv2.namedWindow('origin', cv2.WINDOW_AUTOSIZE)
> # gray 이름으로 창 생성
> cv2.namedWindow('gray', cv2.WINDOW_NORMAL)
> 
> # origin 창에 이미지 표시
> cv2.imshow('origin', img)
> # gray 창에 이미지 표시
> cv2.imshow('gray', img_gray)
> 
> # 창 위치 변경
> cv2.moveWindow('origin', 0, 0)
> cv2.moveWindow('gray', 100, 100)
> 
> # 창 크기 변경
> cv2.waitKey(0)
> cv2.resizeWindow('origin', 200, 200)
> cv2.resizeWindow('gray', 100, 100)
> 
> # gray창 닫기
> cv2.waitKey(0)
> cv2.destroyWindow('gray')
> 
> # 모든 창 닫기
> cv2.waitKey(0)
> cv2.destroyAllWindows()
> ```
>
# 2.2 그림 그리기

그림 그리기는 `객체나 얼굴을 인식해서 그 영역에 사각형을 그려서 표시` 하거나 `이름을 글씨로 표시` 하는 등의 용도로 자주 활용

### 2.2.1 직선 그리기

**cv2.line()** : 직선을 그리는 함수

- cv2.line(img, start, end, color [, thickness, lineType]): 직선 그리기
    - img : 그림 그릴 대상 이밎, Numpy 배열
    - start : 선 시작 지점 좌표(x, y)
    - end : 선 끝 지점 좌표(x, )
    - color : 선 색상, (Blue, Green, Red), 0~255
    - thickness=1 : 선 두께, 생략하면 1픽셀 적용
    - lineType : 선 그리기 형식
        - cv2.LINE_4 : 4 연결 선 알고리즘
        - cv2.LINE_8 : 8 연결 선 알고리즘
        - cv2.LINE_AA : 안티 앨리어싱(antialiasing, 계단 현상 없는 선)

> **[예제 2-10] 다양한 선 그리기(2.10_draw_line.ipynb)**
> 
> 
> ```python
> import cv2
> 
> img = cv2.imread('./img/blank_500.jpg')
> 
> # 파란색 1픽셀 선
> cv2.line(img, (50, 50), (150, 50), (255, 0, 0))
> # 초록색 1픽셀 선
> cv2.line(img, (200, 50), (300, 50), (0, 255, 0))
> # 빨간색 1픽셀 선
> cv2.line(img, (350, 50), (450, 50), (0, 0, 255))
> 
> # 하늘색(파랑+초록) 10픽셀 선
> cv2.line(img, (100, 100), (400, 100), (255, 255, 0), 10)
> # 분홍색(파랑+빨강) 10픽셀 선
> cv2.line(img, (100, 150), (400, 150), (255, 0, 255), 10)
> # 노란색(초록+빨강) 10픽셀 선
> cv2.line(img, (100, 200), (400, 200), (0, 255, 255), 10)
> # 회색(파랑+초록+빨강) 10픽셀 선
> cv2.line(img, (100, 250), (400, 250), (200, 200, 200), 10)
> # 검은색 10픽셀 선
> cv2.line(img, (100, 300), (400, 300), (0, 0, 0), 10)
> 
> # 4연결 선
> cv2.line(img, (100, 350), (400, 400), (0, 0, 255), 20, cv2.LINE_4)
> # 8연결 선
> cv2.line(img, (100, 400), (400, 450), (0, 0, 255), 20, cv2.LINE_8)
> # 안티앨리어싱 선
> cv2.line(img, (100, 450), (400, 500), (0, 0, 255), 20, cv2.LINE_AA)
> # 이미지 전체에 대각선
> cv2.line(img, (0, 0), (500, 500), (0, 0, 255))
> ```
> 
> ![Untitled](https://user-images.githubusercontent.com/69300448/219046566-f960bfc5-1e09-483a-9583-9f76995662e3.png)
> 

### 2.2.2 사각형 그리기

**cv2.rectangle()** : 사각형 그리는 함수

어느 지점이든 시작 지점과 그 반대 지점을 사용한다. 사각형의 크기는 두 좌표의 차이만큼

- **cv2.rectangle(img, start, end, color[, thickness, lineType])** : 사각형 그리기
    - img : 그림 그릴 대상 이미지, Numpy 배열
    - start : 사각형 시작 꼭짓점(x, y)
    - end : 사각형 끝 꼭짓점(x, y)
    - color : 색상(Blue, Green, Red)
    - thickness : 선 두께
        - -1 : 채우기
    - lineType : 선 타입, cv2.line() 과 동일

> **[예제 2-11] 사각형 그리기(2.11_draw_rect.ipynb)**
> 
> 
> ```python
> import cv2
> 
> img = cv2.imread('./img/blank_500.jpg')
> 
> cv2.rectangle(img, (50, 50), (150, 150), (255, 0, 0))
> cv2.rectangle(img, (300, 300), (100, 100), (0, 255, 0), 10)
> cv2.rectangle(img, (450, 200), (200, 450), (0, 0, 255), -1)
> 
> cv2.imshow('rectangle', img)
> cv2.waitKey(0)
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled 1](https://user-images.githubusercontent.com/69300448/219046716-09ac0c32-82d2-4425-9e16-3d76e1f0a16f.png)
> 

### 2.2.3 다각형 그리기

**cv2.polylines()** : 다각형을 그리는 함수

- **cv2.polylines(img, points, isClosed, color[, thickness, lineType])**: 다각형 그리기
    - img : 그림 그릴 대상 이미지
    - points : 꼭짓점 좌표, Numpy 배열 리스트
    - isClosed : 닫힌 도형 여부, True / False
    - color : 색상(Blue, Green, Red)
    - thickness : 선 두께, 채우기 효과(-1)은 지원하지 않음
    - lineType : 선 타입, cv2.line() 과 동일

여러 개의 꼭짓점 좌표를 전달, 전달하는 좌표 형식은 Numpy 배열

> **[예제 2-12] 다각형 그리기(2.12_draw_poly.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> img = cv2.imread('./img/blank_500.jpg')
> 
> # Numpy 배열로 좌표 생성
> # 번개 모양 선 좌표
> pts1 = np.array([[50, 50], [150, 150], [100, 140], [200, 240]], dtype=np.int32)
> # 삼각형 좌표
> pts2 = np.array([[350, 50], [250, 200], [450, 200]], dtype=np.int32)
> pts3 = np.array([[150, 300], [50, 450], [250, 450]], dtype=np.int32)
> # 오각형 좌표
> pts4 = np.array([[350, 250], [450, 350], [400, 450], [300, 450], [250, 350]], 
>                 dtype=np.int32)
> 
> # 다각형 그리기
> cv2.polylines(img, [pts1], False, (255, 0, 0))
> cv2.polylines(img, [pts2], False, (0, 0, 0), 10)
> cv2.polylines(img, [pts3], True, (0, 0, 255), 10)
> cv2.polylines(img, [pts4], True, (0, 0, 0))
> 
> cv2.imshow('polyline', img)
> cv2.waitKey(0)
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled 2](https://user-images.githubusercontent.com/69300448/219046774-d1017ffc-69af-429b-aead-6adb2bb2d95b.png)
> 

### 2.2.4 원, 타원, 호 그리기

- **cv2.circle(img, center, radius, color [, thickness, lineType])** : 원 그리기 함수
    - img : 그림 대상 이미지
    - center : 원점 좌표(x, y)
    - radius : 원의 반지름
    - color : 색상(Blue, Green, Red)
    - thickness : 선 두께(-1 : 채우기)
    - lineType : 선 타입, cv2.line() 과 동일
- **cv2.ellipse(img, center, axes, angle, from, to, color[, thickness, lineType])** : 호나 타원 그리기 함수
    - img : 그림 대상 이미지
    - center : 원점 좌표(x, y)
    - axes : 기준 축 길이
    - angle : 기준 축 회전 각도
    - from, to : 호를 그릴 시작 각도와 끝 각도

> **[예제 2-13] 원, 타원, 호 그리기(2.13_draw_circle.ipynb)**
> 
> 
> ```python
> import cv2
> 
> img = cv2.imread('./img/blank_500.jpg')
> 
> # circle()
> cv2.circle(img, (150, 150), 100, (255, 0, 0))
> cv2.circle(img, (300, 150), 70, (0, 255, 0), 5)
> cv2.circle(img, (400, 150), 50, (0, 0, 255), -1)
> 
> # ellipse()
> # 원점(50, 300), 반지름(50), 회전 0, 0, 도부터 360도
> cv2.ellipse(img, (50, 300), (50, 50), 0, 0, 360, (0, 0, 255))
> # 원점(150, 300), 아래 반원
> cv2.ellipse(img, (150, 300), (50, 50), 0, 0, 180, (255, 0, 0))
> # 원점(200, 300), 위 반원
> cv2.ellipse(img, (200, 300), (50, 50), 0, 181, 360, (0, 0, 255))
> 
> # 원점(325, 300), 반지름(75, 50) 납작한 타원
> cv2.ellipse(img, (325, 300), (75, 50), 0, 0, 360, (0, 255, 0))
> # 원점(450, 300), 반지름(50, 75) 홀쭉한 타원
> cv2.ellipse(img, (450, 300), (50, 75), 0, 0, 360, (255, 0, 255))
> 
> # 원점(50, 425), 반지름(50, 75), 회전 15도
> cv2.ellipse(img, (50, 425), (50, 75), 15, 0, 360, (0, 0, 0))
> # 원점(200, 425), 반지름(50, 75), 회전 45도
> cv2.ellipse(img, (200, 425), (50, 75), 45, 0, 360, (0, 0, 0))
> 
> # 원점(350, 425), 홀쭉한 타원 45도 회전 후 아래 반원 그리기
> cv2.ellipse(img, (350, 425), (50, 75), 45, 0, 180, (0, 0, 255))
> # 원점(400, 425), 홀쭉한 타원 45도 회전 후 위 반원 그리기
> cv2.ellipse(img, (400, 425), (50, 75), 45, 181, 360, (255, 0, 0))
> 
> cv2.imshow('circle', img)
> cv2.waitKey(0)
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled 3](https://user-images.githubusercontent.com/69300448/219046849-4afd5a12-aab1-4d69-9b50-5d671b4782f6.png)
> 

호를 표시하고자 할 때 시작 각도는 3시 방향에서 시작하여 시계 방향으로 돌면서 6시 방향에서 90도, 9시 방향에서 180도와 같은 방식으로 진행

### 2.2.5 글씨 그리기

**cv2.putText()** : 문자열을 이미지에 표시하는 함수

- **cv2.putText(img, text, point, fontFace, fontSize, color [, thichness, lineType])**
    - img : 글씨를 표시할 이미지
    - text : 표시할 문자열
    - point : 글씨를 표시할 좌표(좌측 하단 기준)(x, y)
    - fontFace : 글꼴
        - cv2.FONT_HERSHEY_PLAIN : 산세리프체 작은 글꼴
        - cv2.FONT_HERSHEY_SIMPLEX : 산세리프체 일반 글꼴
        - cv2.FONT_HERSHEY_DUPLEX : 산세리프체 진한 글꼴
        - cv2.FONT_HERSHEY_COMPLEX_SMALL : 세리프체 작은 글꼴
        - cv2.FONT_HERSHEY_COMPLEX : 세리프체 일반 글꼴
        - cv2.FONT_HERSHEY_TRIPLEX : 세리프체 진한 글꼴
        - cv2.FONT_HERSHEY_SCRIPT_SIMPLEX : 필기체 산세리프 글꼴
        - cv2.FONT_HERSHEY_SCRIPT_COMPLEX : 필기체 세리프 글꼴
        - cv2.FONT_ITALIC : 이탤릭체 플래그
    - fontSize : 글꼴 크기
    - color, thickness, lineType : cv2.rectangle() 과 동일

> **[예제 2-14] 글씨 그리기(2.14_draw_text.ipynb)**
> 
> 
> ```python
> import cv
> 
> img = cv2.imread('./img/blank_500.jpg')
> 
> # sans-serif small
> cv2.putText(img, 'Plain', (50, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
> # sans-serif normal
> cv2.putText(img, 'Simplex', (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
> # sans-serif bold
> cv2.putText(img, 'Duplex', (50, 110), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0))
> # sans-serif normal X2
> cv2.putText(img, 'Simplex', (200, 110), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 250))
> 
> # serif small
> cv2.putText(img, 'Complex Small', (50, 180), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0))
> # serif normal
> cv2.putText(img, 'Complex', (50, 220), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
> # serif bold
> cv2.putText(img, 'Triplex', (50, 260), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0))
> # serif normal X2
> cv2.putText(img, 'Complex', (200, 260), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255))
> 
> # hand_written sans-serif
> cv2.putText(img, 'Script Simplex', (50, 330), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 0, 0))
> # hand_written serif
> cv2.putText(img, 'Script Complex', (50, 370), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0, 0, 0))
> 
> # sans-serif + italic
> cv2.putText(img, 'Plain Italic', (50, 430), cv2.FONT_HERSHEY_PLAIN | cv2.FONT_ITALIC, 1, (0, 0, 0))
> # serif + italic
> cv2.putText(img, 'Complex Italic', (50, 470), cv2.FONT_HERSHEY_COMPLEX | cv2.FONT_ITALIC, 1, (0, 0, 0))
> 
> cv2.imshow('draw text', img)
> cv2.waitKey()
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled 4](https://user-images.githubusercontent.com/69300448/219046901-cb71a55c-91e7-43e9-8f2a-5373a875208c.png)
>

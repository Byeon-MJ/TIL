# 2.4 이벤트 처리

키보드와 마우스 입력 방법 처리

### 2.4.1 키보드 이벤트

**cv2.waitKey(delay)** : 키보드 입력 알아내는 함수

delay 인자에 밀리초 단위로 숫자를 전달

해당 시간동안 프로그램을 멈추고 대기하다가 키보드의 눌린 키에 대응하는 코드 값을 정수로 반환한다.

지정한 시간까지 키보드 입력이 없으면 -1을 반환

delay 인자에 0을 전달하면 대기 시간을 무한대로 하겠다는 의미, 키를 누를 때까지 프로그램은 멈춘다.

```python
key = cv2.waitKey(0)
print(key)
```

입력된 키를 특정 문자와 비교할 때는 파이썬 함수인 ord() 함수를 사용하면 편리하다.

키보드의 ‘a’ 키를 눌렀는지 확인하기 위한 코드는 아래와 같다.

```python
if cv2.waitKey(0) == ord('a'):
	pass
```

그런데, 몇몇 64비트 환경에서 cv2.waitKey() 함수는 8비트(ASCII 코드 크기)보다 큰 32비트 정수를 반환한다. 그 값을 ord() 함수를 통해 비교하면 서로 다른 값으로 판단하는 경우 발생

이 때는 하위 8비트를 제외한 비트를 지워야 한다.

0xFF는 하위 8비트가 모두 1로 채워진 숫자이므로 이것과 & 연산을 수행하여 하위 8비트보다 높은 비트는 모두 0으로 채울 수 있다.

```python
key = cv2.waitKey(0) & 0xFF
if key == ord('a'):
	pass
```

> **[예제 2-16] 키 이벤트(2.16_event_key.ipynb)**
> 
> 
> ```python
> import cv2
> 
> img_file = './img/girl.jpg'
> img = cv2.imread(img_file)
> title = 'IMG'
> x, y = 100, 100
> 
> while True:
>     if img is not None:
>         cv2.imshow(title, img)
>         cv2.moveWindow(title, x, y)
>         
>         key = cv2.waitKey(0) & 0xFF  # 키보드 입력 무한대기, 8비트 마스크 처리
>         print(key, chr(key))
>         
>         if key == ord('h'):
>             x -= 10
>         elif key == ord('j'):
>             y += 10
>         elif key == ord('k'):
>             y -= 10
>         elif key == ord('l'):
>             x += 10
>         elif key == ord('q') or key == 27:  # 'q'이거나 'esc'이면 종료
>             cv2.destroyAllWindows()
>             break
>     
>         cv2.moveWindow(title, x, y)   # 새로운 좌표로 창 이동
> ```
> 

### 2.4.2 마우스 이벤트

이벤트를 처리할 함수를 미리 선언 → **cv2.setMouseCallback()** 함수에 그 함수를 전달

```python
def onMouse(event, x, y, flags, param):
	# 여기에 마우스 이벤트에 맞게 해야할 작업을 작성
	pass

cv2.setMouseCallback('title', onMouse)
```

- cv2.setMouseCallback(win_name, onMouse [, param]): onMouse 함수를 등록
    - win_name : 이벤트를 등록할 윈도 이름
    - onMouse : 이벤트 처리를 위해 미리 선언해 놓은 콜백 함수
    - param : 필요에 따라 onMouse 함수에 전달할 인자
- MouseCallback(event, x, y, flags, param) : 콜백 함수 선언부
    - event : 마우스 이벤트 종류, cv2.EVENT_로 시작하는 상수(12가지)
        - cv2.EVENT_MOUSEMOVE : 마우스 움직임
        - cv2.EVENT_LBUTTONDOWN : 왼쪽 버튼 누름
        - cv2.EVENT_RBUTTONDOWN : 오른쪽 버튼 누름
        - cv2.EVENT_MBUTTONDOWN : 가운데 버튼 누름
        - cv2.EVENT_LBUTTONUP : 왼쪽 버튼 뗌
        - cv2.EVENT_RBUTTONUP : 오른쪽 버튼 똄
        - cv2.EVENT_MBUTTONUP : 가운데 버튼 뗌
        - cv2.EVENT_LBUTTONDBLCLK : 왼쪽 버튼 더블 클릭
        - cv2.EVENT_RBUTTONDBLCLK : 오른쪽 버튼 더블 클릭
        - cv2.EVENT_MBUTTONDBLCLK : 가운데 버튼 더블 클릭
        - cv2.EVENT_MOUSEWHEEL : 휠 스크롤
        - cv2.EVENT_MOUSEWHWHEEL : 휠 가로 스크롤
    - x, y : 마우스 좌표
    - flags : 마우스 동작과 함께 일어난 상태, cv2.EVENT_FLAG_로 시작하는 상수(6가지)
        
        어떤 상태인지 알기 위해서는 비트 단위 & (논리 곱) 또는 | (논리합) 연산을 써서 알아냄
        
        - cv2.EVENT_FLAG_LBUTTON(1) : 왼쪽 버튼 누름
        - cv2.EVENT_FLAG_RBUTTON(2) : 오른쪽 버튼 누름
        - cv2.EVENT_FLAG_MBUTTON(4) : 가운데 버튼 누름
        - cv2.EVENT_FLAG_CTRLKEY(8) : Ctrl 키 누름
        - cv2.EVENT_FLAG_SHIFTKEY(16) : Shift 키 누름
        - cv2.EVENT_FLAG_ALTKEY(32) : Alt 키 누름
    - param : cv2.setMouseCallback() 함수에서 전달한 인자

> **[예제 2-17] 마우스 이벤트로 동그라미 그리기(2.17_event_mouse_circle.ipynb)**
> 
> 
> ```python
> import cv2
> 
> def onMouse(event, x, y, flags, param):
>     print(event, x, y, )
>     if event == cv2.EVENT_LBUTTONDOWN:
>         cv2.circle(img, (x, y), 30, (0, 0, 0), -1)
>         cv2.imshow(title, img)
> 
> title = 'mouse event'
> img = cv2.imread('./img/blank_500.jpg')
> if img is not None:
>     cv2.imshow(title, img)
>             
> cv2.setMouseCallback(title, onMouse)  # 마우스 콜백 함수 GUI에 등록
> 
> while True:
>     if cv2.waitKey(0) & 0xFF == 27:  # esc로 종료
>         break
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled](https://user-images.githubusercontent.com/69300448/219047087-f737a971-36b5-4d18-b964-e7689f87215c.png)
> 

이벤트 내에서 그리기를 했다면 반드시 그림이 그려진 이미지를 다시 화면에 표시해야 한다.

이벤트 처리 함수의 선언부는 5개의 인자를 모두 선언해야 한다.

함수 내부에서 사용하지 않더라도 5개의 인자는 모두 선언부에 기재해야 하며, 그렇지 않으면 오류가 발생

> **[예제 2-18] 플래그를 이용한 동그라미 그리기(2.18_event_mouse_circle_flag.ipynb)**
> 
> 
> ```python
> import cv2
> 
> def onMouse(event, x, y, flags, param):
>     print(event, x, y, flags)   # 파라미터 출력
>     color = colors['black']
>     if event == cv2.EVENT_LBUTTONDOWN:  # 왼쪽 버튼 누른 경우
>         # 컨트롤 + 시프트 둘다
>         if flags & cv2.EVENT_FLAG_CTRLKEY and flags & cv2.EVENT_FLAG_SHIFTKEY:
>             color = colors['green']
>         # 시프트 키만
>         elif flags & cv2.EVENT_FLAG_SHIFTKEY:
>             color = colors['blue']
>         # 컨트롤 키만
>         elif flags & cv2.EVENT_FLAG_CTRLKEY:
>             color = colors['red']
>         # 알트 키만
>         elif flags & cv2.EVENT_FLAG_ALTKEY:
>             color = colors['white']
>             
>         # 지름 30크기의 원을 해당 좌표에 그림
>         cv2.circle(img, (x, y), 30, color, -1)
>         cv2.imshow(title, img)
> 
> title = 'mouse event'
> img = cv2.imread('./img/blank_500.jpg')
> colors = {'black' : (0, 0, 0),
>          'red' : (0, 0, 255),
>          'green' : (0, 255, 0),
>          'blue' : (255, 0, 0),
>          'white' : (255, 255, 255)}
> 
> if img is not None:
>     cv2.imshow(title, img)
> else:
>     print("No read Image")
>     
> cv2.setMouseCallback(title, onMouse)
> 
> while True:
>     if cv2.waitKey(0) == ord('c'):      # c 키 입력하면 클리어
>         img = cv2.imread('./img/blank_500.jpg')
>         cv2.imshow(title, img)
>     elif cv2.waitKey(0) & 0xFF == 27:  # esc로 종료
>         break
> 
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled 1](https://user-images.githubusercontent.com/69300448/219047150-9274290b-fea6-4f0c-8cc3-b0e0f760deb4.png)
> 

### 2.4.3 트랙바

트랙바(track-bar)는 슬라이드 모양의 인터페이스를 마우스로 움직여서 값을 입력받는 GUI 요소

**cv2.createTrack()** 함수로 생성, 마우스 이벤트 방식과 마찬가지로 트랙바를 움직였을 때 동작할 함수를 미리 준비해서 함께 전달한다.

트랙바의 값을 얻기 위한 **cv2.getTrackbarPos()** 함수도 함께 쓰인다.

```python
def onChange(value):
v = cv2.getTrackbarPos('trackbar', 'win_name')

cv2.createTrackbar('trackbar', 'win_name', 0, 100, onChange)
```

- **cv2.createTrackbar(trackbar_name, win_name, value, count, onChange)** : 트랙바 생성
    - trackbar_name : 트랙바 이름
    - win_name : 트랙바를 표시할 창 이름
    - value : 트랙바 초기 값, 0~count 사이의 값
    - count : 트랙바 눈금의 개수, 트랙바가 표시할 수 있는 최대 값
    - onChange : TrackbarCallback, 트랙바 이벤트 핸들러 함수
- **TrackbarCallback(value)** : 트랙바 이벤트 콜백 함수
    - value : 트랙바가 움직인 새 위치 값
- **pos = cv2.getTrackbarPos(trackbar_name, win_name)**
    - trackbar_name : 찾고자 하는 트랙바 이름
    - win_name : 트랙바가 있는 창의 이름
    - pos : 트랙바 위치 값

> **[예제 2-19] 트랙바를 이용한 이미지 색 조정(2.19_event_trackbar.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> win_name = 'Trackbar'
> img = cv2.imread('./img/blank_500.jpg')
> 
> # 트랙바 이벤트 처리 함수
> def onChange(x):
>     print(x)
>     
>     # 'R', 'G', 'B' 트랙바 위치 값
>     r = cv2.getTrackbarPos('R', win_name)
>     g = cv2.getTrackbarPos('G', win_name)
>     b = cv2.getTrackbarPos('B', win_name)
>     print(r, g, b)
>     
>     img[:] = [b, g, r]
>     cv2.imshow(win_name, img)
> 
> cv2.imshow(win_name, img)
> 
> # 트랙바 생성
> cv2.createTrackbar('R', win_name, 0, 255, onChange)
> cv2.createTrackbar('G', win_name, 0, 255, onChange)
> cv2.createTrackbar('B', win_name, 0, 255, onChange)
> 
> while True:
>     if cv2.waitKey(1) & 0xFF == 27:
>         break
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled 2](https://user-images.githubusercontent.com/69300448/219047194-be7e381f-067c-490f-aea7-f231f1ba42dd.png)
> 
1. 이벤트 처리 함수를 만들고
2. 트랙바를 생성하면서 전달
3. 트랙바를 움직이면, 함수의 인자를 통해 얻을 수 있게 됨
4. cv2.getTrackbarPos() 함수에 트랙바 이름과 트랙바가 위치한 창의 이름을 지정해서 원하는 트랙바의 새로운 값을 얻을 수 있다.

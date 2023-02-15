# 2.1 이미지와 비디오 입출력

OpenCV를 이용한 대부분의 작업은 파일로 된 이미지를 읽어서 적절한 연산을 적용하고 그 결과를 화면에 표시하거나 다른 파일로 저장하는 것

### 2.1.1 이미지 읽기

- **img = cv2.imread(file_name [, mode_flag])** : 파일로부터 이미지 읽기
    - file_name : 이미지 경로, 문자열
    - mode_flag = cv2.IMREAD_COLOR : 읽기 모드 지정
        - cv2.IMREAD_COLOR : 컬러(BGR) 스케일로 읽기, 기본 값
        - cv2.IMREAD_UNCHANGED : 파일 그대로 읽기
        - cv2.IMREAD_GRAYSCALE : 그레이(흑백) 스케일로 읽기
    - img : 읽은 이미지, Numpy 배열
    - cv2.imshow(title, img) : 이미지를 화면에 표시
    - title : 창 제목, 문자열
    - img : 표시할 이밎, Numpy 배열
- **Key = cv2.waitKey([delay])** : 키보드 입력 대기
    - delay = 0 : 키보드 입력을 대기할 시간(ms), 0 : 무한대(기본 값)
    - key : 사용자가 입력한 키 값, 정수
        - -1 : 대기시간 동안 키 입력 없음

> **[예제 2-1] 이미지 파일을 화면에 표시(2.1_img_show.ipynb)**
> 
> 
> ```python
> import cv2
> 
> img_file = './img/girl.jpg'
> img = cv2.imread(img_file)
> 
> if img is not None:
>     cv2.imshow('IMG', img)
>     cv2.waitKey()
>     cv2.destroyAllWindows()
> else:
>     print('No image file.')
> ```
> 

**cv2.imread()** : 파일로부터 이미지를 읽을 때 모드를 지정할 수 있다.

모드를 지정하지 않으면 3개 채널(B, G, R)로 구성된 컬러 스케일로 읽어들임

필요에 따라 그레이 스케일 또는 파일에 저장된 스케일 그대로 읽을 수 있다.

```python
img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
```

> **[예제 2-2] 이미지 파일을 그레이 스케일로 화면에 표시(2.2_img_show_gray.ipynb)**
> 
> 
> ```python
> import cv2
> 
> img_file = './img/girl.jpg'
> img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
> 
> if img is not None:
>     cv2.imshow('IMG', img)
>     cv2.waitKey()
>     cv2.destroyAllWindows()
> else:
>     print('No image file.')
> ```
> 

### 2.1.2 이미지 저장하기

읽어들인 이미지를 다시 파일로 저장하는 함수는 cv2.imwrite()

- **cv2.imwrite(file_path, img)** : 이미지를 파일에 저장
    - file_path : 저장할 파일 경로 이름, 문자열
    - img : 저장할 영상, Numpy 배열

> **[예제 2-3] 컬러 이미지를 그레이 스케일로 저장(2.3_img_wirte_grayscale.ipynb)**
> 
> 
> ```python
> import cv2
> 
> img_file = './img/girl.jpg'
> save_file_name = './result/girl_gray.jpg'
> 
> img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
> 
> if img is not None:
>     cv2.imshow(img_file, img)
>     cv2.imwrite(save_file_name, img)
>     cv2.waitKey()
>     cv2.destroyAllWindows()
> else:
>     print('No Image File.')
> ```
> 

### 2.1.3 동영상 및 카메라 프레임 읽기

OpenCV는 동영상 파일이나 컴퓨터에 연결한 카메라 장치로부터 연속된 이미지 프레임을 읽을 수 있는 API를 제공

- **cap = cv2.VideoCapture(file_path 또는 index)** : 비디오 캡처 객체 생성자
    - file_path : 동영상 파일 경로
    - index : 카메라 장치 번호, 0부터 순차적으로 증가(0, 1, 2, …)
    - cap : VideoCapture 객체
- **ret = cap.isOpened()** : 객체 초기화 확인
    - ret : 초기화 여부, True / False
- **ret, img = cap.read()** : 영상 프레임 읽기
    - ret : 프레임 읽기 성공 또는 실패 여부, Treu / False
    - img : 프레임 이미지, Numpy 배열 또는 None
- **cap.set(id, value)** : 프로퍼티 변경
- **cap.get(id)** : 프로퍼티 확인
- **cap.release()** : 캡처 자원 반납

**cv2.VideoCapture()** : 생성자 함수를 사용하여 영상 프레임을 읽는 객체를 생성

동영상 파일 경로 이름을 전달하면 동영상 파일에 저장된 프레임을 읽을 수 있다.

카메라 장치 번호를 전달하면 카메라로 촬영하는 프레임을 읽을 수 있다.

**isOpened()** : 파일이나 카메라 장치에 제대로 연결되었는지 확인

**read()** : 다음 프레임을 읽을 수 있다.

Boolean과 Numpy 배열 객체를 쌍으로 갖는 튜플 (ret, img) 객체를 반환

ret 값이 True이면 다음 프레임 읽기에 성공한 것이고, img를 꺼내서 사용하면 됨

**set(), get()** : 여러가지 속성을 얻거나 지정 가능

**release()** : 프로그램을 종료하기 전에 호출해서 자원을 반납

### 2.1.4 동영상 파일 읽기

> **[예제 2-4] 동영상 파일 재생(2.4_video_play.ipynb)**
> 
> 
> ```python
> import cv2
> 
> video_file = './img/big_buck.avi'
> 
> cap = cv2.VideoCapture(video_file)
> 
> if cap.isOpened():
>     while True:
>         ret, frame = cap.read()
>         if ret:
>             cv2.imshow(video_file, frame)
>             cv2.waitKey(25)
>         else:
>             break
> else:
>     print("can't open video.")
>     
> cap.release()
> cv2.destroyAllWindows()
> ```
> 

cv2.waitKey(25) 가 필요한 이유는 각 프레임을 화면에 표시하는 시간이 너무 빠르면 우리 눈으로 볼수 없기 때문에 지연 시간을 조정하여 적절한 속도로 영상을 재생하게 해준다.

### 2.1.5 카메라(웹캠) 프레임 읽기

**cv2.VideoCapture()** : 카메라로 프레임을 읽기 위해서는 카메라 장치 인덱스 번호를 정수로 지정

카메라 장치 인덱스는 0부터 시작해서 1씩 증가

나머지는 동영상 파일과 다루는 법이 거의 똑같다.

> **[예제 2-5] 카메라 프레임 읽기(2.5_video_cam.ipynb)**
> 
> 
> ```python
> import cv2
> 
> cap = cv2.VideoCapture(0)
> if cap.isOpened():
>     while True:
>         ret, frame = cap.read()
>         if ret:
>             cv2.imshow('camera', frame)
>             if cv2.waitKey(1) != -1:
>                 break
>         else:
>             print('no frame')
>             break
> else:
>     print("can't open camera.")
> cap.release()
> cv2.destroyAllWindows()
> ```
> 

카메라로부터 프레임을 읽는 경우 파일의 끝이 정해져 있지 않으므로 무한루프를 빠져 나올 조건이 필요하다.

**cv2.waitKey()** : 지정한 대기 시간 동안 키 입력이 없으면 -1을 반환

반환된 값이 -1이 아니면 아무 키나 입력되었다는 뜻이고 이때 break를 걸어줄 수 있음

```python
if cv2.waitKey(1) != -1:
	break
```

### 2.1.6 카메라 비디오 속성 제어

캡처 객체에는 영상 또는 카메라의 여러가지 속성을 확인하고 설정할 수 있는 get(id), set(id, value) 함수를 제공

속성을 나타내는 아이디는 cv2.CAP_PROP_FRAME_로 시작하는 상수로 정의되어 있다.

자세한 속성에 대한 명세는 API 문서를 참조하기 바란다.

- 속성 ID : ‘cv2.CAP_PROP_’로 시작하는 상수
    - cv2.CAP_PROP_FRAME_WIDTH : 프레임 폭
    - cv2.CAP_PROP_FRAME_HEIGHT : 프레임 높이
    - cv2.CAP_PROP_FPS : 초당 프레임 수
    - cv2.CAP_PROP_POS_MSEC : 동영상 파일의 프레임 위치(ms)
    - cv2.CAP_PROP_POS_AVI_RATIO : 동영상 파일의 상대 위치(0: 시작, 1: 끝)
    - cv2.CAP_PROP_FOURCC : 동영상 파일 코덱 문자
    - cv2.CAP_PROP_AUTOFOCUS : 카메라 자동 초점 조절
    - cv2.CAP_PROP_ZOOM : 카메라 줌

FPS를 구하는 상수를 활용하여 동영상의 FPS를 구하고 적절한 지연시간을 계산해서 지정할 수 있다.

지연 시간은 밀리초 단위이고, 정수만 전달할 수 있다.

```python
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000/fps)
```

> **[예제 2-6] FPS를 지정해서 동영상 재생(2.6_video_play_fps.ipynb)**
> 
> 
> ```python
> import cv2
> 
> video_file = './img/big_buck.avi'
>  
> cap = cv2.VideoCapture(video_file)
> 
> if cap.isOpened():
>     fps = cap.get(cv2.CAP_PROP_FPS)
>     delay = int(1000/fps)
>     print(f'FPS:{fps}, Delay: {delay}')
> 
>     while True:
>         ret, img = cap.read()
>         if ret:
> #             흑백 영상 보기
> #             img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
>             cv2.imshow(video_file, img)
>             cv2.waitKey(delay)
>         else:
>             break
> else:
>     print("can't open video.")
> 
> cap.release()
> cv2.destroyAllWindows()
> ```
> 

카메라로부터 읽은 영상이 너무 고화질인 경우 픽셀 수가 많아 연산하는데 시간이 많이 걸리는 경우가 있다.

이때 프레임의 폭과 높이를 제어해서 픽셀 수를 줄일 수 있다. (카메라가 아닌 동영상 파일에 프레임 크기를 재지정 할 수는 없다.)

cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT

> **[예제 2-7] 카메라 프레임 크기 설정(2.7_video_cam_resize.ipynb)**
> 
> 
> ```python
> import cv2
> 
> cap = cv2.VideoCapture(0)
> width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)            # 프레임 폭 값 구하기
> height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)          # 프레임 높이 값 구하기
> print(f'Original width: {width}, height: {height}')
> 
> cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)               # 프레임 폭을 320으로 설정
> cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)              # 프레임 높이를 240으로 설정
> width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
> height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
> print(f'Resized width: {width}, height: {height}')
> 
> if cap.isOpened():
>     while True:
>         ret, frame = cap.read()
>         if ret:
>             cv2.imshow('camera', frame)
>             if cv2.waitKey(1) != -1:
>                 break
>         else:
>             print('no frame!')
>             break
> else:
>     print("can't open camera")
>     
> cap.release()
> cv2.destroyAllWindows()
> ```
> 

### 2.1.7 비디오 파일 저장하기

카메라로부터 프레임을 표시하다가 특정 키(혹은 아무 키)를 누르면 해당 프레임을 파일로 저장하는 코드

> **[예제 2-8] 카메라로 사진 찍기(2.8_video_cam_take_pic.ipynb)**
> 
> 
> ```python
> import cv2
> 
> cap = cv2.VideoCapture(0)
> cnt_num = 0
> if cap.isOpened():
>     while True:
>         ret, frame = cap.read()
>         if ret:
>             cv2.imshow('camera', frame)
> #             if cv2.waitKey(1) != -1:
> #                 cv2.imwrite('photo.jpg', frame)
> #                 break
>             if cv2.waitKey(1) == ord('q'):
>                 break
>             elif cv2.waitKey(1) == ord('a'):
>                 cv2.imwrite(f'./result/photo{cnt_num}.jpg', frame)                
>                 cnt_num += 1
>         else:
>             print('no frame!')
>             break
> else:
>     print('no camera!')
> cap.release()
> cv2.destroyAllWindows()
> ```
> 

여러 프레임을 동영상으로 저장하려고 할 때는 cv2.VideoWriter() 라는 새로운 API가 필요

- writer = cv2.VideoWriter(file_path, fourcc, fps, (width, height)) : 비디오 저장 클래스 생성자 함수
    - file_path : 비디오 파일 저장 경로
    - fourcc : 비디오 인코딩 형식 4글자
    - fps : 초당 프레임 수
    - (width, height) : 프레임 폭과 프레임 높이
    - writer : 생성된 비디오 저장 객체
- writer.write(frame) : 프레임 저장
    - frame : 저장할 프레임, Numpy 배열
- writer.set(id, value) : 프로퍼티 변경
- writer.get(id) : 프로퍼티 확인
- ret = writer.fourcc(c1, c2, c3, c4) : fourcc 코드 생성
    - c1, c2, c3, c4 : 인코딩 형식 4글자, ‘MJPG’, ‘DIVX’ 등
    - ret : fourcc 코드
- cv2.VideoWriter_fourcc(c1, c2, c3, c4) : cv2.VideoWriter.fourcc()와 동일

**cv2.VideoWriter()** : 저장할 파일 이름과 인코딩 포맷 문자, fps, 프레임 크기를 지정해서 객체를 생성

**write()** : 프레임을 파일에 저장

**cv2.VideoWriter_fourcc()** : 4개의 인코딩 포맷 문자를 전달하면 코드 값을 생성해 내는 함수

사용할 수 있는 인코딩 형식 문자 확인 : https://fourcc.org/codecs.php

```python
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
```

> **[예제 2-9] 카메라로 녹화하기(2.9_video_cam_rec.ipynb)**
> 
> 
> ```python
> import cv2
> 
> cap = cv2.VideoCapture(0)
> if cap.isOpened:
>     file_path = './result/record.avi'
>     fps = 25.40
>     
>     # 인코딩 포맷 문자
>     fourcc = cv2.VideoWriter_fourcc(*'DIVX')
>     
>     width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
>     height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
>     size = (int(width), int(height))
>     out = cv2.VideoWriter(file_path, fourcc, fps, size)
>     
>     while True:
>         ret, frame = cap.read()
>         if ret:
>             cv2.imshow('camera-recoding', frame)
> #             out.write(frame)
> #             if cv2.waitKey(int(1000/fps)) != -1:
> #                 break
>             
>             # 'a'를 입력할때만 녹화
>             if cv2.waitKey(25) == ord('a'):
>                 out.write(frame)
>             
>             # 'q'를 입력하면 종료
>             if cv2.waitKey(1) == ord('q'):
>                 break
>         
>         else:
>             print('no frame!')
>             break
>     out.release()
> else:
>     print("can't open camera!")
> cap.release()
> cv2.destroyAllWindows()
> ```
>
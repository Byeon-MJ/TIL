# Can't parse 'center'. Sequence item with index 0 has a wrong type

OpenCV 를 이용하여 Face alignment 를 진행하는 도중에 발생한 오류

- Github 코드
    
    [Computer_Vision_Project/Face_Alignment.ipynb at main · Byeon-MJ/Computer_Vision_Project](https://github.com/Byeon-MJ/Computer_Vision_Project/blob/main/Face_Alignment.ipynb)
    

두 눈을 중앙점으로 두 눈 사이의 중앙점을 구해서 회전하기위해 getRotationMatrix2D 메소드를 사용하였는데 아래 에러가 발생하였다.

```python
rotate = cv2.getRotationMatrix2D(eyes_center, degree, scale)

TypeError: Can't parse 'center'. Sequence item with index 0 has a wrong type
```

stackoverflow 를 찾아보니 좌표값을 int값으로 넣어주면 된다고 하는데…

[OpenCV cv2.circle "can't parse 'center'" error](https://stackoverflow.com/questions/67594458/opencv-cv2-circle-cant-parse-center-error)

분명 중앙점을 설정할때 int로 타입을 바꿔주었고

```python
right_eye_center = np.mean(points[RIGHT_EYE], axis = 0).astype('int')
left_eye_center = np.mean(points[LEFT_EYE], axis = 0).astype('int')
```

eyes_center의 Type도 확인을 했는데 int 타입이었다.

```python
# 두 눈의 중앙점 간의 중앙점을 다시 계산
eyes_center = ((left_eye_center[0, 0] + right_eye_center[0, 0]) // 2,
                   (left_eye_center[0, 1] + right_eye_center[0, 1]) // 2)
```

```python
print(eyes_center)
print(type(eyes_center[0]), type(eyes_center[1]))

(414, 191)
<class 'numpy.int64'> <class 'numpy.int64'>
```

### Solution

```python
eyes_center = (int((left_eye_center[0, 0] + right_eye_center[0, 0]) // 2),
                   int((left_eye_center[0, 1] + right_eye_center[0, 1]) // 2))

(414, 191)
<class 'int'> <class 'int'>
```

getRotationMatrix2D 메소드에 들어가는 eyes_center 변수의 값을 직접적으로 int type으로 형변환을 시켜주니까 해결되었다.

numpy.int64 는 오류가 나고 python ‘Int’ 로 들어가야만 하는가보다!!

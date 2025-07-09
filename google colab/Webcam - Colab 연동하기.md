# Webcam - Colab 연동하기

Webcam을 이용한 프로젝트를 진행할때, Colab에서 OpenCV의 VideoCapture(0)를 사용하지 못해서 매번 JupyterNotebook을 사용했었다.

Colab에서는 JavaScript를 이용하여 Webcam을 사용할 수 있다고 해서 찾아보았다.

처음에 찾았던 조금 이전 버전내용은 마지막에 Old version으로 따로 붙여놓았다. 

지금 Colab에서 사용하는 코드는 전체적으로 비슷했지만 비동기방식 등이 적용되어 조금 더 개선되었다.

## Colab - Webcam 연동

아래 내용은 기본적인 Colab에서 Webcam을 연동하는 방법이다.

- advancde_outputs  Colab Notebook
    
    [Google Colaboratory](https://colab.research.google.com/notebooks/snippets/advanced_outputs.ipynb#scrollTo=2viqYx97hPMi)
    

### 코드 스니펫에서 Camera Capture 불러오기

Colab에서 왼쪽 아래 < > 모양을 누르면 미리 설정된 코드를 불러올 수 있는 코드 스니펫을 불러올 수 있다.

![Untitled](https://user-images.githubusercontent.com/69300448/217141777-2859f854-155f-487e-adec-049723efe49e.png)

![Untitled 1](https://user-images.githubusercontent.com/69300448/217141800-d6e456c8-a056-45f6-b156-ce6110221cd2.png)

```python
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode

def take_photo(filename='photo.jpg', quality=0.8):
  js = Javascript('''
    async function takePhoto(quality) {
      const div = document.createElement('div');
      const capture = document.createElement('button');
      capture.textContent = 'Capture';
      div.appendChild(capture);

      const video = document.createElement('video');
      video.style.display = 'block';
      const stream = await navigator.mediaDevices.getUserMedia({video: true});

      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();

      // Resize the output to fit the video element.
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

      // Wait for Capture to be clicked.
      await new Promise((resolve) => capture.onclick = resolve);

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      stream.getVideoTracks()[0].stop();
      div.remove();
      return canvas.toDataURL('image/jpeg', quality);
    }
    ''')
  display(js)
  data = eval_js('takePhoto({})'.format(quality))
  binary = b64decode(data.split(',')[1])
  with open(filename, 'wb') as f:
    f.write(binary)
  return filename
```

아래 코드를 실행하면 Camera가 실행되고 Capture 버튼을 누르면 사진을 저장할 수있다!

```python
from IPython.display import Image
try:
  filename = take_photo()
  print('Saved to {}'.format(filename))
  
  # Show the image which was just taken.
  display(Image(filename))
except Exception as err:
  # Errors will be thrown if the user does not have a webcam or if they do not
  # grant the page permission to access it.
  print(str(err))
```

## Old Version

[Google Colaboratory](https://colab.research.google.com/drive/1tbAeRge6KKgCYdC6ihDrsl80aRYoVOMa#scrollTo=T7tY2g3ATPo1&forceEdit=true&sandboxMode=true%EB%A5%BC)

```python
from IPython.display import HTML, Image
from google.colab.output import eval_js
from base64 import b64decode

VIDEO_HTML = """
<video autoplay
 width=800 height=600></video>
<script>
var video = document.querySelector('video')
navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream=> video.srcObject = stream)
  
var data = new Promise(resolve=>{
  video.onclick = ()=>{
    var canvas = document.createElement('canvas')
    var [w,h] = [video.offsetWidth, video.offsetHeight]
    canvas.width = w
    canvas.height = h
    canvas.getContext('2d')
          .drawImage(video, 0, 0, w, h)
    video.srcObject.getVideoTracks()[0].stop()
    video.replaceWith(canvas)
    resolve(canvas.toDataURL('image/jpeg', %f))
  }
})
</script>
"""

def take_photo(filename='photo.jpg', quality=0.8):
  display(HTML(VIDEO_HTML % quality))
  data = eval_js("data")
  binary = b64decode(data.split(',')[1])
  with open(filename, 'wb') as f:
    f.write(binary)
  return len(binary)
```

```python
take_photo()
```

이제 Colab에서도 Webcam을 사용할 수 있으니, OpenPose를 이용한 포즈 추출이나 

이전에 작업했던(아직 후기를 작성해서 올리진 못했지만…) Mask Detection 모델 들도 적용해볼 수 있을 것 같다!

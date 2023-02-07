# [Colab] Webcam - Colab 연동하기

Webcam을 이용한 프로젝트를 진행할때, Colab에서 OpenCV의 VideoCapture(0)를 사용하지 못해서 매번 JupyterNotebook을 사용했었다.

Colab에서는 JavaScript를 이용하여 Webcam을 사용할 수 있다고 해서 한번 찾아보고 사용해보았다.

아래 내용은 기본적은 Colab 가이드 문서에서 Webcam을 연동하는 방법이다.

- Colab : <a href="https://colab.research.google.com/github/Byeon-MJ/TIL/blob/main/%5BColab%5D%20Webcam_Colab_Interlock.ipynb"><img data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" src="https://camo.githubusercontent.com/84f0493939e0c4de4e6dbe113251b4bfb5353e57134ffd9fcab6b8714514d4d1/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667"></a>

• Reference : Colab 가이드 문서

 [https://colab.research.google.com/drive/1tbAeRge6KKgCYdC6ihDrsl80aRYoVOMa#scrollTo=T7tY2g3ATPo1&forceEdit=true&sandboxMode=true%EB%A5%BC](https://colab.research.google.com/drive/1tbAeRge6KKgCYdC6ihDrsl80aRYoVOMa#scrollTo=T7tY2g3ATPo1&forceEdit=true&sandboxMode=true%EB%A5%BC)

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

# [ML] KNN, Scaling

[Notion 정리 링크](https://stump-python-602.notion.site/ML-KNN-Scaling-Regression-Linear-Polynomial-Multiple-Regularization-27add895f71e4e0b9788a4409b4912e2)
# K-최근접 이웃(KNN)

- 지도 학습 알고리즘 중 하나, 굉장히 직관적이고 간단하다.
- 어떤 데이터가 주어지면 그 주변(이웃)의 데이터를 살펴본 뒤 더 많은 데이터가 포함되어 있는 범주로 분류하는 방식
- KNN은 분류와 회귀 문제 둘다 사용할 수 있다.

## KNN Classification

![Untitled](https://user-images.githubusercontent.com/69300448/209594695-eff4642a-3302-47d5-995d-cebac08f034b.png)

출처: towardsdatascience

- 새로운 데이터(빨간 점) 이 주어졌을 때 이를 A로 분류할지, B로 분류할지 판단하는 문제
- k=3(안쪽 원)일 때, 노란색 점(Class A) 1개와 보라색 점(Class B) 2개가 있다. 따라서 k=3일 때 빨간 점은 B로 분류 된다.
- k=6(바깥쪽 원)일 때, 노란색 점 4개와 보라색 점 2개가 있으므로 노란색 점으로 분류된다.
- K의 값에 따라 결과가 달라지므로 너무 작아서도 안되고 너무 커서도 안된다.
- 보통 K 값은 홀수를 사용한다. 짝수일 경우 동점이 발생할 수가 있어 결과를 도출할 수 없기 때문이다.

## 거리 계산

- KNN 에서는 데이터와 데이터 사이의 거리에 따라 결과가 달라질 수 있다.
- 데이터의 스케일에 따라 큰 영향을 받기 때문에 데이터 전처리가 필요함

### 1. Euclidean Distance

일반적인 점과 점사이의 거리를 구하는 방법이다.

![Untitled 1](https://user-images.githubusercontent.com/69300448/209594792-9fe3b4be-3511-4158-85e7-cb1c5707ba95.png)

출처: Wikipedia

3차원에서 유클리드 거리 구하는 예시

![Untitled 2](https://user-images.githubusercontent.com/69300448/209594871-b9c1994d-183b-48a2-afcf-a68a16de3297.png)

출처: ratsgo's blog

![Untitled 3](https://user-images.githubusercontent.com/69300448/209594950-d120e051-5987-444d-b384-c6e22363ea73.png)

### 2. Manhattan Distance

점과 점 사이의 직선 거리가 아니라 X축과 Yㅊ푹을 따라 간 거리를 의미한다.

아래 예시의 Route 1, Route 2, Route 3은 서로 다른 길이지만 맨해튼 거리로는 같다.

![Untitled 4](https://user-images.githubusercontent.com/69300448/209595100-5cab0aa7-7c8c-495a-9ecd-1dc1449fa09a.png)

## KNN Regression

- KNN 회귀 모델은 분류와 동일하게 가까운 k개의 이웃을 찾는다.
- 그 다음 이웃 샘플 타깃값의 평균을 구하여 예측값으로 사용한다.
- Sckikit Learn은 회귀 모델의 점수로 결정계수 값을 반환한다.

# Data Scaling

- 데이터 간의 거리에 영향을 받는 알고리즘을 사용할 때, 데이터의 스케일을 일정한 기준으로 맞춰 주는 방법
- 크게 Standardizaion(표준화) 와 Normalization(정규화) 가 있음
    
    

## Standardization

- 특성들의 평균을 0, 분산을 1로 스케일링

### z-Score Standardization(Numpy)

- 표준점수를 활용해 Numpy로 표준화하기

```python
mean = mp.mean(train_input, axis=0)
std = mp.std(train_imput, axis=0)

train_scaled = (train_input - mean) / std
```

## Normalization

- 특성들을 특정 범위(주로 0~1)로 스케일링 하는 것

## Sickit Learn Scaler

- scaler 는 fit 과 transform 메서드를 가진다
- fit 메서드는 훈련 데이터에 적용해서 훈련 데이터의 분포를 먼저 학습
- tranform 메서드를 이용해 훈련 데이터와 테스트 데이터에 적용
- fit 과 transform이 결합된 fit_transform() 메서드가 있다. 훈련데이터에만 적용할 것
- 특성에 따라 각기 다른 스케일링을 적용하는 것이 유리할 수 있음

### StandardScaler()

- 특성들의 평균을 0, 분산을 1로 스케일링
- 최솟값과 최댓값의 크기를 제한하지 않기때문에 특정 알고리즘에서는 문제가 있을 수 있음
- **이상치에 매우 민감**
- 회귀보다는 분류에 유용하다

### MinMaxScaler()

- 가장 작은 값이 0, 가장 큰 값이 1로 변환되므로, 모든 특성들이 [0, 1]의 범위를 갖게 스케일링
- **이상치에 매우 민감**
- 분류보다 회귀에 유용하다

### ****MaxAbsScaler()****

- 각 특성의 절대값이 0과 1사이가 되도록 스케일링
- 모든 값이 -1 ~ 1의 값으로 표현된다
- 데이터가 모두 양수일 경우에는 MinMaxScaler와 같다.
- **이상치에 매우 민감**

### ****RobustScaler()****

- 평균과 분산 대신에 중앙값과 사분위값을 사용
- **이상치의 영향을 최소화**

### Normalizer()

- Normalizer 의 경우 각 샘플(행)마다 적용되는 방식
- 한 행의 모든 특성들 사이의 유클리드 거리가 1이 되도록 스케일링
- 일반적인 데이터 전처리의 상황에서 사용되는 것이 아니라 모델(특히나 딥러닝) 내 학습 벡터에 적용


## Reference
https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing<br>
https://justweon-dev.tistory.com/19<br>

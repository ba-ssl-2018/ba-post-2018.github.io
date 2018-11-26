---
title: Distance/Reconstrunction-based Novelty Detections
categories:
- General
excerpt: |
  Distance/Reconstrunction-based Novelty Detections
feature_text: |
  ## Distance/Reconstrunction-based Novelty Detections
  <br><br/>
  이 글은 고려대학교 강필성 교수님의 Business Analytics 강의를 참조하였습니다.
feature_image: "https://picsum.photos/2560/600?image=733"
image: "https://picsum.photos/2560/600?image=733"
use_math: true
---





<br><br><br><br/><br/><br/>
<h2> k-Nearest Neighbor-based Novelty Detection </h2>

각각의 데이터 포인트마다, 자신을 제외한 인접한 k개의 데이터 포인트와의 거리를 계량함으로써 novelty score을 계산하고, <br/>
이를 근거로 outlier를 걸러내는 것을 골조로 하는 방식입니다. <br/>
이 때, 거리를 계량하는 방식에 따라
* max 거리
<img src="/images/1_max.png" width="1800" height="600" />
* average 거리
<img src="/images/2_average.png" width="1800" height="600" />
* mean 거리
<img src="/images/3_mean.png" width="1800" height="600" />
등으로 세분화 할 수 있습니다.



<h4> Average 거리를 이용한 코드예시 </h4>

```python
from sklearn.datasets import load_digits
import numpy as np
## 1 ############################################################################################
def k_neighbor_novelty_score(from_point, to_points, k):
    if len(to_points) != k:
        raise
    all_dists = 0
    for each_to_point in to_points:
        each_dist = np.sqrt(np.sum((from_point - each_to_point)**2))
        all_dists += each_dist
    novelty_score = np.round(all_dists/k)
    return novelty_score
## 2 ############################################################################################
input_digits, target_labels = load_digits(n_class=10, return_X_y=True)
num_data = 101
k = 100
input_digits = input_digits[:num_data]
target_labels = target_labels[:num_data]
novelty_scores = [0]*num_data
for outer_idx in np.arange(num_data):
    novelty_scores[outer_idx] = k_neighbor_novelty_score(input_digits[outer_idx], [element for inner_idx, element in enumerate(input_digits) if inner_idx != outer_idx], k)
## 3 ############################################################################################
print(novelty_scores)
# Read data and set some threshold
threshold = 50
outliers = [idx for idx, element in enumerate(novelty_scores) if element > threshold]
print(outliers)
```



<h4> 코드 설명 </h4>

1. def k_neighbor_novelty_score(from_point, to_points): <br/>
함수는 한 점과(from_point), 그 점을 제외한 k개의 점을(to_points) 인풋으로 받아 <br/>
from_point-to_points간의 평균 거리를 바탕으로, from_point의 to_points에 대한 novelty score을 반환합니다.
2. 101개의 숫자데이터(각각이 pixels를 담고 있는 record)를 로드하고, <br/>
각각의 데이터를 하나하나 looping하면서 각 데이터별, 자신을 제외한 k=100 이웃과의 novelty score을 계산하여 순서대로 쌓습니다.
3. 누적된 novelty score을 뽑아보고 임의의 임계값을 정하여 outliers를 골라냅니다.





<br><br><br><br/><br/><br/>
<h2> Clustering-based Novelty Detection </h2>

각각의 데이터 포인트마다, 가장 가까운 centroid까지의 거리를 계량함으로써 novelty score을 계산하고, <br/>
이를 근거로 outlier를 걸러내는 것을 골조로 하는 방식입니다(centroid란, k-means clustering, 즉, EM 알고리즘의 수행 결과로써 나온 centrods를 의미합니다). <br/>
이 때, 거리를 계량하는 방식에 따라
* 절대 거리
* 상대 거리
로 세분화 될 수 있습니다.



<h4> 절대 거리를 이용한 코드예시 </h4>

```python
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
import numpy as np
## 1 ############################################################################################
def closest_centroid_novelty_score(from_point, to_centroids):
    all_dists = [0]*len(to_centroids)
    for each_centroid_idx, each_centroid in enumerate(to_centroids):
        each_dist = np.sqrt(np.sum((from_point - each_centroid)**2))
        all_dists[each_centroid_idx] = each_dist
    novelty_score = np.min(all_dists)
    return novelty_score
## 2 ############################################################################################
input_digits, target_labels = load_digits(n_class=10, return_X_y=True)
num_data = 1000
input_digits = input_digits[:num_data]
target_labels = target_labels[:num_data]
k_means_output_obj = KMeans(n_clusters=10, n_jobs=4).fit(input_digits)
centroids = k_means_output_obj.cluster_centers_
novelty_scores = [0]*num_data
for outer_idx in np.arange(num_data):
    novelty_scores[outer_idx] = closest_centroid_novelty_score(input_digits[outer_idx], centroids)
## 3 ############################################################################################
print(novelty_scores)
# Read data and set some threshold
threshold = 35
outliers = [idx for idx, element in enumerate(novelty_scores) if element > threshold]
print(outliers)
```



<h4> 코드 설명 </h4>

1. def closest_centroid_novelty_score(from_point, to_centroids): <br/>
함수는 한 점과(from_point), k-means의 결과로써 나온 centroids(to_centroids)를 인풋으로 받아 <br/>
from_point-to_centroids L2-norm을 바탕으로, from_point에서 각각의 centroid까지의 절대 거리를 구하여 쌓고, 그 중에서 최솟값을 찾아 novelty score로써 반환합니다.
2. 1000개의 숫자데이터(각각이 pixels를 담고 있는 record)를 로드하고, k-means clustering 시킵니다. <br/>
이 때, 숫자 라벨의 종류는 10가지(0~9) 이므로 총 10개의 clusters가 생기도록 합니다. <br/>
각각의 데이터를 하나하나 looping하면서 각 데이터별, 10개의 centroids에 대한 novelty score을 계산하여 순서대로 쌓습니다.
3. 누적된 novelty score을 뽑아보고 임의의 임계값을 정하여 outliers를 골라냅니다.





<br><br><br><br/><br/><br/>
<h2> Principal Component Analysis-based Novelty Detection </h2>

각각의 데이터 포인트마다, PCA를 하기 전의 데이터 포인트 X와, PCA를 수행한 후 다시 reconstruct한 데이터 포인트 X' 간의 절대 거리를 계량함으로써 novelty score을 계산하고, <br/>
이를 근거로 outlier를 걸러내는 것을 골조로 하는 방식입니다. <br/>
이 때,
* PCA를 하기 전의 데이터 공간을 X 공간: span by original data representing basis of x_1 and x_2 (holding original data points X)
* PCA에 의하여 projected된 공간을 Z 공간: span by princiapl compoenent basis of z_1 and z_2 (reserving maximum projected variance)
<img src="/images/4_to_z.png" width="1800" height="600" />
* PCA이후 reconstructed된 데이터 공간을 x공간: span by original data representing basis of x_1 and x_2 (holding reconstructed data points X')
<img src="/images/4_back_to_x.png" width="1800" height="600" />
이라 칭하겠습니다.



<h4> 절대 거리를 이용한 코드예시 </h4>

```python
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import numpy as np
## 1 ############################################################################################
def reconstruction_loss_novelty_score(from_points, to_reconstructed_points):
    novelty_scores = np.sqrt(np.sum((from_points - to_reconstructed_points)**2, axis=1))
    return novelty_scores
## 2 ############################################################################################
input_digits, target_labels = load_digits(n_class=10, return_X_y=True)
num_data = 100
input_digits = input_digits[:num_data]
target_labels = target_labels[:num_data]
pca_obj = PCA()
projected_input_digits = pca_obj.fit_transform(input_digits)
reconstructed_input_digits = pca_obj.inverse_transform(projected_input_digits)
novelty_scores = reconstruction_loss_novelty_score(input_digits, reconstructed_input_digits)
## 3 ############################################################################################
print(novelty_scores)
# Read data and set some threshold
threshold = 9e-14
outliers = [idx for idx, element in enumerate(novelty_scores) if element > threshold]
print(outliers)
```



<h4> 코드 설명 </h4>

1. def reconstruction_loss_novelty_score(from_points, to_reconstructed_points): <br/>
함수는 PCA를 하기 전 모든 데이터 포인트와(from_points), PCA를 수행한 후 다시 reconstruct한 모든 데이터 포인트(to_reconstructed_points)를 인풋으로 받아 <br/>
from_points-to_reconstructed_points L2-norm을 바탕으로, <br/>
from_points의 각 점에서 to_reconstructed_points의 각 점까지의 절대 거리를 구하여, novelty score로써 반환합니다.
2. 100개의 숫자데이터(각각이 pixels를 담고 있는 record)를 로드하고, PCA와 reconstruction을 수행합니다. <br/>
이어 데이터 포인트 전체와 reconstructed된 데이터 포인트 전체를 상기 함수에 feeding함으로써, <br/>
전체 데이터 포인트 각각의 요소에 corresponding하는 reconstructed 데이터 포인트에 대한 novelty score을 계산합니다.
3. novelty score을 뽑아보고 임의의 임계값을 정하여 outliers를 골라냅니다.

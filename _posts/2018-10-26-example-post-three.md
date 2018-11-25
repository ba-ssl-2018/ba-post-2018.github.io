---
title: Distance/Reconstrunction-based Novelty Detections
categories:
- General
excerpt: |
  Distance/Reconstrunction-based Novelty Detections
feature_text: |
  Distance/Reconstrunction-based Novelty Detections <br><br/>
  이 글은 고려대학교 강필성 교수님의 Business Analytics 강의를 참조하였습니다.
feature_image: "https://picsum.photos/2560/600?image=733"
image: "https://picsum.photos/2560/600?image=733"
use_math: true
---





<br><br><br><br/><br/><br/>
<h2> k-Nearest Neighbor-based Novelty Detection </h2>

각각의 데이터 포인트마다, 자신을 제외한 인접한 k개의 데이터 포인트와의 거리를 계량함으로써 novelty score을 계산하고, 이를 근거로 outlier를 걸러내는 것을 골조로 하는 방식입니다. 이 때, 거리를 계량하는 방식에 따라
* max 거리
{% include figure.html image="/images/1_max.png" width="300"%}
* average 거리
{% include figure.html image="/images/2_average.png" width="300"%}
* mean 거리
{% include figure.html image="/images/3_mean.png" width="300"%}
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

1. def k_neighbor_novelty_score(from_point, to_points):
함수는 한 점과(from_point), 그 점을 제외한 k개의 점을(to_points) 인풋으로 받아
from_point-to_points간의 평균 거리를 바탕으로, from_point의 to_points에 대한 novelty score을 반환합니다.
2. 101개의 숫자데이터(각각이 pixels를 담고 있는 record)를 로드하고, 각각의 데이터를 하나하나 looping하면서 각 데이터별, 자신을 제외한 k=100 이웃과의 novelty score을 계산하여 순서대로 쌓습니다.
3. 누적된 novelty score을 뽑아보고 임의의 임계값을 정하여 outliers를 골라냅니다.





<br><br><br><br/><br/><br/>
<h2> Clustering-based Novelty Detection </h2>

각각의 데이터 포인트마다, 가장 가까운 centroid까지의 거리를 계량함으로써 novelty score을 계산하고, 이를 근거로 outlier를 걸러내는 것을 골조로 하는 방식입니다(centroid란, k-means clustering, 즉, EM 알고리즘의 수행 결과로써 나온 centrods를 의미합니다).
이 떄, 거리를 계량하는 방식에 따라
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

1. def closest_centroid_novelty_score(from_point, to_centroids):
함수는 한 점과(from_point), k-means의 결과로써 나온 centroids(to_centroids)를 인풋으로 받아
from_point-to_centroids L2-norm을 바탕으로, from_point에서 각각의 centroid까지의 절대 거리를 구하여 쌓고, 그 중에서 최솟값을 찾아 novelty score로써 반환합니다.
2. 1000개의 숫자데이터(각각이 pixels를 담고 있는 record)를 로드하고, k-means clustering 시킵니다. 이 떄 숫자 라벨의 종류는 10가지(0~9) 이므로, 총 10개의 clusters가 생기도록 합니다.
각각의 데이터를 하나하나 looping하면서 각 데이터별, 10개의 centroids에 대한 novelty score을 계산하여 순서대로 쌓습니다.
3. 누적된 novelty score을 뽑아보고 임의의 임계값을 정하여 outliers를 골라냅니다.

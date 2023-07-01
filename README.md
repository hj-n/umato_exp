모든 파일의 실행은 `conda activate umato`로 가상환경 활성화 후 이루어져야 한다.

### scalability.py
실행: `umato_exp` 폴더 하에서 `python3.11 scalability.py`
기능: 모든 92개의 dataset에 대해(inf가 발생하지 않는 dataset은 90개) 각 알고리즘의 수행 시간을 dataset마다 5번씩 측정해 그 평균을 csv 파일로 기록, `scalability/scalability.csv`에 저장

### scalability_subsample.py
실행: `umato_exp` 폴더 하에서 `python3.11 scalability_subsample.py`
기능: 크기가 10000 이상인 7개의 dataset과 각 알고리즘에 대해 각 dataset의 subsample을 각 크기별로(100, 200, 500, 1000, 2000, 5000, 10000) 만들어 그를 기반으로 수행 시간을 측정하는 것을 5번 반복, 그 평균을 csv 파일로 기록, `scalability/scalability_subsample.csv`에 저장

### analysis.py
실행: `umato_exp/scalability` 폴더 하에서 `python3.11 analysis.py`
기능: `scalability.csv`의 데이터를 기반으로 각 알고리즘의 수행 시간 통계량을 뽑고, 수행 시간의 boxplot을 그려 `boxplot.png`로 저장
비활성화한 기능: dimension-time 그래프, length-time 그래프를 UMATO와 UMAP에 대해서 그림

### analysis_subsample.py
실행: `umato_exp/scalability` 폴더 하에서 `python3.11 analysis_subsample.py`
기능: `scalability_subsample.csv`의 데이터를 기반으로 각 알고리즘의 subsample 크기에 따른 평균 수행 시간을 뽑고, 수행 시간의 plot을 그려 `scalability_subsample.png`로 저장
비활성화한 기능: dimension-time 그래프, length-time 그래프를 UMATO와 UMAP에 대해서 그림

### 주의사항
실험을 진행하였을 때는 `scalability.py`, `scalability_subsample.py`를 모두 `umato_exp`의 부모 폴더에, `analysis.py`, `analysis_subsample.py`는 그 부모 폴더의 하위에 있는 `scalability` 폴더에 넣은 뒤 진행하였기 때문에 바뀐 폴더 위치에서는 실험을 진행하는 두 코드에서 `ImportError`가 발생할 수 있습니다. 바뀐 위치에 맞게 내부에서 사용되는 경로를 수정하였지만, 특히 UMATO를 import할 때 오류가 발생할 수 있습니다. 그 경우 import 경로를 적절하게 수정하면 됩니다.
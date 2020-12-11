import cv2
import numpy as np

src = cv2.imread("Image/apple_books.jpg")
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
target = cv2.imread("Image/apple.jpg", cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create(
    nfeatures=40000,
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=31,
    firstLevel=0,
    WTA_K=2,
    scoreType=cv2.ORB_HARRIS_SCORE,
    patchSize=31,
    fastThreshold=20,
)
# ORB 클래스(cv2.ORB_create)로 ORB 객체를 생성합니다.
# cv2.ORB_create(최대 피처 수, 스케일 계수, 피라미드 레벨, 엣지 임곗값, 시작 피라미드 레벨, 비교점, 점수 방식, 패치 크기, FAST 임곗값)을 의미합니다.
# 최대 피처 수는 ORB 객체가 한 번에 검출하고자 하는 특징점의 개수 입니다.
# 스케일 계수는 이미지 피라미드를 설정합니다. 인수를 2로 지정할 경우, 이미지 크기가 절반이 되는 고전적인 이미지 피라미드를 의미합니다.
# 스케일 계수를 너무 크게 지정하면 특징점의 매칭 확률을 떨어뜨립니다. 반대로 스케일 계수를 적게 지정하면 더 많은 피라미드 레벨을 구성해야 하므로 연산 속도가 느려집니다.
# 피라미드 레벨은 이미지 피라미드의 레벨 수를 나타냅니다.
# 엣지 임곗값은 이미지 테두리에서 발생하는 특징점을 무시하기 위한 경계의 크기를 나타냅니다.
# 시작 피라미드 레벨은 원본 이미지를 넣을 피라미드의 레벨을 의미합니다.
# 비교점은 BRIEF 기술자가 구성하는 비교 비트를 나타냅니다.
# 2를 지정할 경우 이진 형식(0, 1)을 사용하며, 3의 값을 사용할 경우 3자 간 비교 결과로 (0, 1, 2)를 사용한다. 4의 값을 사용할 경우 4자 간 비교 결과로 (0, 1, 2, 3)을 사용합니다.
# 이 매개변수에는 2(1비트), 3(2비트), 4(2비트)의 값만 지정해 비교할 수 있습니다.
# 점수 방식은 피처의 순위를 매기는 데 사용되며, 해리스 코너(cv2.ORB_HARRIS_SCORE) 방식과 FAST(cv2.ORB_FAST_SCORE) 방식을 사용할 수 있습니다.
# 패치 크기는 방향성을 갖는 BFIEF 기술자가 사용하는 개별 피처의 패치 크기입니다.
# 패치 크기는 엣지 임곗값 매개변수와 상호작용하므로 패치 크기의 값을 변경한다면 엣지 임곗값이 패치 크기의 값보다 커야 합니다.
# FAST 임곗값은 FAST 검출기에서 사용되는 임곗값을 의미합니다.


kp1, des1 = orb.detectAndCompute(gray, None)
kp2, des2 = orb.detectAndCompute(target, None)
# 각각의 이미지에 특징점 및 기술자 계산 메서드(orb.detectAndCompute)로 특징점 및 기술자를 계산합니다.
# 특징점, 기술자 = orb.detectAndCompute(입력 이미지, 마스크)을 의미합니다.
# 특징점은 좌표(pt), 지름(size), 각도(angle), 응답(response), 옥타브(octave), 클래스 ID(class_id)를 포함합니다.
# 좌표는 특징점의 위치를 알려주며, 지름은 특징점의 주변 영역을 의미합니다.
# 각도는 특징점의 방향이며, -1일 경우 방향이 없음을 나타냅니다.
# 응답은 피처가 존재할 확률로 해석하며, 옥타브는 특징점을 추출한 피라미드의 스케일을 의미합니다.
# 클래스 ID는 특징점에 대한 저장공간을 생성할 때 객체를 구분하기 위한 클러스터링한 객체 ID를 뜻합니다.
# 기술자는 각 특징점을 설명하기 위한 2차원 배열로 표현됩니다. 이 배열은 두 특징점이 같은지 판단할 때 사용됩니다.


bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# 특징점과 기술자 검출이 완료되면, 전수 조사 매칭(Brute force matching)을 활용해 객체를 인식하거나 추적할 수 있습니다.
# 전수 조사란 관심의 대상이 되는 집단을 이루는 모든 개체를 조사해서 모집단의 특성을 측정하는 방법입니다.
# 전수 조사 매칭은 객체의 이미지와 객체가 포함된 이미지의 각 특징점을 모두 찾아 기술자를 활용하는 방식입니다.
# 이때 가장 우수한 매칭을 판단하기 위해 유효 거리를 측정합니다. 유효 거리가 짧을수록 우수한 매칭입니다.
# 그러므로, 전수 조사 매칭 클래스(cv2.BFMatcher)로 전수 조사 매칭을 사용합니다.
# orb.detectAndCompute(거리 측정법, 교차 검사)을 의미합니다.
# 거리 측정법은 질의 기술자(Query Descriptors)와 훈련 기술자(Train Descriptors)를 비교할 때 사용되는 거리 계산 측정법을 지정합니다.
# 질의(Query)와 훈련(Train)이라는 용어로 인해 마치 추론 모델을 만드는 것처럼 착각할 수 있습니다.
# 질의는 객체를 탐지할 이미지를 뜻하며, 훈련은 질의 공간에서 검출할 요소를 의미한다고 볼 수 있습니다.
# 여기서 훈련은 객체로 인식된 이미지를 탐지할 수 있도록 사전(Dictionary)이라는 공간에 포함하는 과정을 말합니다.
# 교차 검사는 훈련된 집합에서 질의 집합이 가장 가까운 이웃이며, 질의 집합에서 훈련된 집합이 가장 가까운 이웃이면 서로 매칭됩니다.

matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
# 매치 함수(orb.match)로 최적의 매칭을 검출합니다.
# bf.detectAndCompute(기술자1, 기술자2)을 의미합니다.
# 질의 기술자(queryDescriptors)와 훈련 기술자(trainDescriptors)를 사용해 최적의 매칭을 찾습니다.
# 기술자 공간에서 작동하는 마스크(mask)의 행은 질의 기술자의 행과 대응하며, 열은 내부 사전 이미지(훈련 기술자)와 대응합니다.
# 반환값으로 DMatch(Dictionary Match) 객체를 반환하며, 4개의 멤버를 갖고 있습니다.
# DMatch 객체는 질의 색인(queryIdx), 훈련 색인(trainIdx), 이미지 색인(imgIdx), 거리(distance)로 구성돼 있습니다.
# 질의 색인과 훈련 색인은 두 이미지의 특징점에서 서로 매칭하기 위해 식별되는 색인 값을 의미합니다.
# 이미지 색인은 이미지와 사전 사이에서 매칭된 경우 훈련에 사용된 이미지를 구별하는 색인값을 의미합니다.
# 거리는 각 특징점 간 유클리드 거리 또는 매칭의 품질을 의미합니다. 거리 값이 낮을수록 매칭이 정확합니다.
# 그러므로, 정렬 함수(sorted)로 거리 값이 낮은 순으로 정렬합니다.

count = 100

for i in matches[:count]:
    idx = i.queryIdx
    x1, y1 = kp1[idx].pt
    cv2.circle(src, (int(x1), int(y1)), 3, (255, 0, 0), 3)
# 반복문을 통해, 우수한 상위 100개에 대해서만 표시합니다.
# 객체가 포함된 이미지에 관한 색인은 멤버 중 질의 색인(queryIdx)에 포함돼 있습니다.
# 이 값을 특징점의 좌표(pt)에 해당하는 질의 색인값을 넣어 지점으로 반환합니다.
# •Tip : 객체 이미지에서 찾는 경우, 훈련 색인(trainIdx)을 불러와 객체 이미지 특징점의 좌표(pt)로 반환합니다.

flag = (cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS | cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
matching = cv2.drawMatches(src, kp1, target, kp2, matches[:count], None, flags=flag)

src = cv2.resize(src, dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)    # 이미지 크기 축소
matching = cv2.resize(matching, dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)    # 이미지 크기 축소

cv2.imshow("src", src)
cv2.imshow("match", matching)

cv2.waitKey()
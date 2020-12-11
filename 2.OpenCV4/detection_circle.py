import cv2

src = cv2.imread("Image/colorball.jpg")
dst = src.copy()
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 200, param1 = 250, param2 = 10, minRadius = 80, maxRadius = 120)
# cv2.HoughCircles(검출 이미지, 검출 방법, 해상도 비율, 최소 거리, 캐니 엣지 임곗값, 중심 임곗값, 최소 반지름, 최대 반지름)를 이용하여 원 검출을 진행합니다.
# 검출 방법은 항상 2단계 허프 변환 방법(21HT, 그레이디언트)만 사용합니다.
# 해상도 비율은 원의 중심을 검출하는 데 사용되는 누산 평면의 해상도를 의미합니다.
# 인수를 1로 지정할 경우 입력한 이미지와 동일한 해상도를 가집니다. 즉, 입력 이미지 너비와 높이가 동일한 누산 평면이 생성됩니다.
# 또한 인수를 2로 지정하면 누산 평면의 해상도가 절반으로 줄어 입력 이미지의 크기와 반비례합니다.
# 최소 거리는 일차적으로 검출된 원 중심과 다음 원 중심 사이의 최소 거리입니다. 이 값은 원이 여러 개 검출되는 것을 줄이는 역할을 합니다.
# 캐니 엣지 임곗값은 허프 변환에서 자체적으로 캐니 엣지를 적용하게 되는데, 이때 사용되는 상위 임곗값을 의미합니다.
# 하위 임곗값은 자동으로 할당되며, 상위 임곗값의 절반에 해당하는 값을 사용합니다.
# 중심 임곗값은 그레이디언트 방법에 적용된 중심 히스토그램(누산 평면)에 대한 임곗값입니다. 이 값이 낮을 경우 더 많은 원이 검출됩니다.
# 최소 반지름과 최대 반지름은 검출될 원의 반지름 범위입니다. 0을 입력할 경우 검출할 수 있는 반지름에 제한 조건을 두지 않습니다.
# 최소 반지름과 최대 반지름에 각각 0을 입력할 경우 반지름을 고려하지 않고 검출하며, 최대 반지름에 음수를 입력할 경우 검출된 원의 중심만 반환합니다.


for i in circles[0]:
    cv2.circle(dst, (i[0], i[1]), int(i[2]), (255, 255, 255), 5)
# 검출을 통해 반환되는 circles 변수는 (1, N, 3)차원 형태를 갖습니다.
# 내부 차원의 요소로는 검출된 중심점(x, y)과 반지름(r)이 저장돼 있습니다.
# 반복문을 활용해 circles 배열에서 중심점과 반지름을 반환할 수 있습니다.
# 검출된 정보는 소수점을 포함합니다. 원 그리기 함수는 소수점이 포함되어도 사용할 수 있으므로, 형변환을 진행하지 않습니다.

src = cv2.resize(src, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR) 
gray = cv2.resize(gray, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR) 
dst = cv2.resize(dst, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR) 

cv2.imshow("src", src)
cv2.imshow("gray", gray)
cv2.imshow("dst", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

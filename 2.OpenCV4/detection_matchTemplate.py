import cv2

src = cv2.imread("Image/hats.png", cv2.IMREAD_GRAYSCALE)
templit = cv2.imread("Image/hat.png", cv2.IMREAD_GRAYSCALE)
dst = cv2.imread("Image/hats.png")

result = cv2.matchTemplate(src, templit, cv2.TM_SQDIFF_NORMED)
# 템플릿 매칭 함수(cv2.matchTemplate)로 템플릿 매칭을 적용합니다.
# cv2.matchTemplate(원본 이미지, 템플릿 이미지, 템플릿 매칭 플래그)을 의미합니다.
# 원본 이미지와 템플릿 이미지는 8비트의 단일 채널 이미지를 사용합니다.
# 템플릿 매칭 플래그는 템플릿 매칭에 사용할 연산 방법을 설정합니다.
# 반환되는 결괏값(dst)은 32비트의 단일 채널 이미지로 반환됩니다.
# 또한, 배열의 크기는 W - w + 1, H - h + 1의 크기를 갖습니다.
# (W, H)는 원본 이미지의 크기이며, (w, h)는 템플릿 이미지의 크기입니다.
# 결괏값이 위와 같은 크기를 갖는 이유는 원본 이미지에서 템플릿 이미지를 일일히 비교하기 때문입니다.
# 예를 들어, 4×4 크기의 원본 이미지와 3×3 크기의 템플릿 이미지가 있다면 아래의 그림과 같이 표현할 수 있습니다.
# 총 4번의 비교를 진행할 수 있으며, 이를 배열로 옮긴다면 2×2 크기를 갖게 됩니다.
# 수식으로 다시 표현한다면, (W−w+1,H−h+1)=(4−3+1,4−3+1)=(2,2) 가 됩니다.

minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
# 결괏값(dst)에서 가장 유사한 부분을 찾기 위해 최소/최대 위치 함수(cv2.minMaxLoc)로 검출값을 찾습니다.
# 최소/최대 위치 함수는 최소 포인터, 최대 포인터, 최소 지점, 최대 지점을 반환합니다.
# 검출 위치의 좌측 상단 모서리 좌표는 최소 지점(minLoc)이나 최대 지점(maxLoc)에 위치합니다.
# 템플릿 이미지를 일일히 비교하므로, 이미지 크기는 템플릿 이미지와 동일합니다.

# •Tip : cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED는 최소 지점(minLoc)이 검출된 위치입니다.
# •Tip : cv2.TM_CCORR, cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED는 최대 지점(maxLoc)이 검출된 위치입니다.

x, y = minLoc
h, w = templit.shape

dst = cv2.rectangle(dst, (x, y), (x +  w, y + h) , (0, 0, 255), 1)
dst = cv2.resize(dst, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)    # 이미지 크기 축소

cv2.imshow("dst", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
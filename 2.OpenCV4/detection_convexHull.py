import cv2

src = cv2.imread("Image/convex.png")
dst = src.copy()

gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
ret, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

for i in contours:
    hull = cv2.convexHull(i, clockwise=True)
    cv2.drawContours(dst, [hull], 0, (0, 0, 255), 2)
# cv2.convexHull()를 활용해 윤곽선에서 블록 껍질을 검출합니다.
# cv2.convexHull(윤곽선, 방향)을 의미합니다.
# 윤곽선은 윤곽선 검출 함수에서 반환되는 구조를 사용합니다.
# 방향은 검출된 볼록 껍질의 볼록점들의 인덱스 순서를 의미합니다.
# 블록 껍질 함수는 단일 형태에서만 검출이 가능합니다.
# 그러므로, 반복문을 활용해 단일 형태의 윤곽선 구조에서 블록 껍질을 검출합니다.
# •Tip : 윤곽선 구조는 윤곽선 검출 함수의 반환값과 형태가 동일하다면, 임의의 배열에서도 검출이 가능합니다.
# •Tip : 방향이 True라면 시계 방향, False라면 반시계 방향으로 정렬됩니다.

cv2.imshow("dst", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
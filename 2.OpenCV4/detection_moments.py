import cv2

src = cv2.imread("Image/convex.png", cv2.IMREAD_COLOR)
dst = src.copy()

gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
ret, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

for i in contours:
    M = cv2.moments(i)
    cX = int(M['m10'] / M['m00'])
    cY = int(M['m01'] / M['m00'])
    cv2.circle(dst, (cX, cY), 3, (255, 0, 0), -1)
    cv2.drawContours(dst, [i], 0, (0, 0, 255), 2)
    # cv2.moments()를 활용해 윤곽선에서 모멘트를 계산합니다.
    # cv2.moments(배열, 이진화 이미지)을 의미합니다.
    # 배열은 윤곽선 검출 함수에서 반환되는 구조 또는 이미지를 사용합니다.
    # 이진화 이미지는 입력된 배열 매개변수가 이미지일 경우, 이미지의 픽셀 값들을 이진화 처리할지 결정합니다.
    # 이진화 이미지 매개변수에 참 값을 할당한다면 이미지의 픽셀 값이 0이 아닌 값은 모두 1의 값으로 변경해 모멘트를 계산합니다.
    # 모멘트 함수를 통해 면적, 평균, 분산 등을 간단하게 구할 수 있습니다.
    

cv2.imshow("dst", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
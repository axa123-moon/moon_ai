import numpy as np
import cv2

src = cv2.imread("Image/harvest.jpg", cv2.IMREAD_COLOR)
height, width, channel = src.shape
print(src.shape)
src_part = src[200:500, 200:500] # 이미지의 특정 영역만 가져옴 y좌표 시작 : y좌표 종료, x좌표 시작 : x좌표 종료
print(src_part.shape)

srcPoint=np.array([[300, 200], [400, 200], [500, 500], [200, 500]], dtype=np.float32)
dstPoint=np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
# 원본 이미지에서 4점 변환할 srcPoint와 결과 이미지의 위치가 될 dstPoint를 선언합니다.
# 좌표의 순서는 좌상, 우상, 우하, 좌하 순서입니다. numpy 형태로 선언하며, 좌표의 순서는 원본 순서와 결과 순서가 동일해야합니다.

matrix = cv2.getPerspectiveTransform(srcPoint, dstPoint)
# 기하학적 변환을 위하여 cv2.getPerspectiveTransform(원본 좌표 순서, 결과 좌표 순서)를 사용하여 matrix를 생성합니다.

dst = cv2.warpPerspective(src, matrix, (width, height))
# cv2.warpPerspective(원본 이미지, 매트릭스, (결과 이미지 너비, 결과 이미지 높이))를 사용하여 이미지를 변환할 수 있습니다.
# 저장된 매트릭스 값을 사용하여 이미지를 변환합니다.
# 이외에도, 보간법, 픽셀 외삽법을 추가적인 파라미터로 사용할 수 있습니다.

dst_part = cv2.resize(dst, dsize=(0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)    # 이미지 크기 축소

pts1 = np.array([[100, 0], [200, 0], [300, 300], [0, 300]])   # 다각형 좌표
cv2.polylines(src_part, [pts1], True, (0, 255, 255), 2)      # 다각형 그리기

cv2.imshow("src_part", src_part)
cv2.imshow("dst_part", dst_part)
#cv2.imshow("dst", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
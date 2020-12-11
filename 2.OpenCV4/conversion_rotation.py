import cv2

src = cv2.imread("Image/ara.jpg", cv2.IMREAD_COLOR)

height, width, channel = src.shape
matrix = cv2.getRotationMatrix2D((width/2, height/2), 90, 1)
# matrix에 회전 배열을 생성하여 저장합니다.
# cv2.getRotationMatrix2D((중심점 X좌표, 중심점 Y좌표), 각도, 스케일)을 설정합니다.
# 중심점은 Tuple형태로 사용하며 회전할 기준점을 설정합니다.
# 각도는 회전할 각도를 설정합니다.
# 스케일은 이미지의 확대 비율을 설정합니다.

dst = cv2.warpAffine(src, matrix, (width, height))
# 결과 이미지로 사용할 dst를 선언하고 회전 함수를 적용합니다.
# cv2.warpAffine(원본 이미지, 배열, (결과 이미지 너비, 결과 이미지 높이))을 의미합니다.
# 결과 이미지의 너비와 높이로 크기가 선언되며 배열에 따라 이미지가 회전합니다.

cv2.imshow("src", src)
cv2.imshow("dst", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
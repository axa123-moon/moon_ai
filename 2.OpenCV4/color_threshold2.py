import cv2

src = cv2.imread("Image/geese.jpg", cv2.IMREAD_COLOR)

gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
dst = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 5)
#dst = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, -5)
# blockSize = 25, (25*25 영역분석하여 임계값 설정)
# 상수값 = 5, 음수일 경우 전체영역이 어두워지고, 값이 높을 경우 전체영역이 밝아짐
# cv2.THRESH_BINARY 임계값 이상 = 최댓값 임계값 이하 = 0 
# cv2.THRESH_BINARY_INV 임계값 이상 = 0 임계값 이하 = 최댓값 
# cv2.THRESH_TRUNC 임계값 이상 = 임계값 임계값 이하 = 원본값 
# cv2.THRESH_TOZERO 임계값 이상 = 원본값 임계값 이하 = 0 
# cv2.THRESH_TOZERO_INV 임계값 이상 = 0 임계값 이하 = 원본값 
# cv2.THRESH_MASK 흑색 이미지로 변경 
# cv2.THRESH_OTSU Otsu 알고리즘 사용 
# cv2.THRESH_TRIANGLE Triangle 알고리즘 사용 

cv2.imshow("dst", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
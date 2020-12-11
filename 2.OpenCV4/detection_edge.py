import cv2

src = cv2.imread("Image/wheat.jpg", cv2.IMREAD_COLOR)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

canny = cv2.Canny(src, 100, 255)
# cv2.Canny(원본 이미지, 임계값1, 임계값2, 커널 크기, L2그라디언트)를 이용하여 가장자리 검출을 적용합니다.
# 임계값1은 임계값1 이하에 포함된 가장자리는 가장자리에서 제외합니다.
# 임계값2는 임계값2 이상에 포함된 가장자리는 가장자리로 간주합니다.
# 커널 크기는 Sobel 마스크의 Aperture Size를 의미합니다. 포함하지 않을 경우, 자동으로 할당됩니다.
# L2그라디언트는 L2방식의 사용 유/무를 설정합니다. 사용하지 않을 경우, 자동적으로 L1그라디언트 방식을 사용합니다.

sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, 3)
# cv2.Sobel(그레이스케일 이미지, 정밀도, x방향 미분, y방향 미분, 커널, 배율, 델타, 픽셀 외삽법)를 이용하여 가장자리 검출을 적용합니다.
# 정밀도는 결과 이미지의 이미지 정밀도를 의미합니다. 정밀도에 따라 결과물이 달라질 수 있습니다.
# x 방향 미분은 이미지에서 x 방향으로 미분할 값을 설정합니다.
# y 방향 미분은 이미지에서 y 방향으로 미분할 값을 설정합니다.
# 커널은 소벨 커널의 크기를 설정합니다. 1, 3, 5, 7의 값을 사용합니다.
# 배율은 계산된 미분 값에 대한 배율값입니다.
# 델타는 계산전 미분 값에 대한 추가값입니다.
# 픽셀 외삽법은 이미지를 가장자리 처리할 경우, 영역 밖의 픽셀은 추정해서 값을 할당해야합니다.
# 이미지 밖의 픽셀을 외삽하는데 사용되는 테두리 모드입니다. 외삽 방식을 설정합니다.
# •Tip : x방향 미분 값과 y방향의 미분 값의 합이 1 이상이여야 하며 각각의 값은 0보다 커야합니다.

laplacian = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
# cv2.Laplacian(그레이스케일 이미지, 정밀도, 커널, 배율, 델타, 픽셀 외삽법)를 이용하여 가장자리 검출을 적용합니다.
# 정밀도는 결과 이미지의 이미지 정밀도를 의미합니다. 정밀도에 따라 결과물이 달라질 수 있습니다.
# 커널은 2차 미분 필터의 크기를 설정합니다. 1, 3, 5, 7의 값을 사용합니다.
# 배율은 계산된 미분 값에 대한 배율값입니다.
# 델타는 계산전 미분 값에 대한 추가값입니다.
# 픽셀 외삽법은 이미지를 가장자리 처리할 경우, 영역 밖의 픽셀은 추정해서 값을 할당해야합니다.
# 이미지 밖의 픽셀을 외삽하는데 사용되는 테두리 모드입니다. 외삽 방식을 설정합니다.
# •Tip : 커널의 값이 1일 경우, 3x3 Aperture Size를 사용합니다. (중심값 = -4)

# 픽셀 외삽법 종류
# cv2.BORDER_CONSTANT iiiiii | abcdefgh | iiiiiii 
# cv2.BORDER_REPLICATE aaaaaa | abcdefgh | hhhhhhh 
# cv2.BORDER_REFLECT fedcba | abcdefgh | hgfedcb 
# cv2.BORDER_WRAP cdefgh | abcdefgh | abcdefg 
# cv2.BORDER_REFLECT_101 gfedcb | abcdefgh | gfedcba 
# cv2.BORDER_REFLECT101 gfedcb | abcdefgh | gfedcba 
# cv2.BORDER_DEFAULT gfedcb | abcdefgh | gfedcba 
# cv2.BORDER_TRANSPARENT uvwxyz | abcdefgh | ijklmno 
# cv2.BORDER_ISOLATED 관심 영역 (ROI) 밖은 고려하지 않음 

src2 = cv2.resize(src, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR) 
canny2 = cv2.resize(canny, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR) 
sobel2 = cv2.resize(sobel, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR) 
laplacian2 = cv2.resize(laplacian, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR) 

cv2.imshow("src", src2)
cv2.imshow("canny", canny2)
cv2.imshow("sobel", sobel2)
cv2.imshow("laplacian", laplacian2)
cv2.waitKey(0)
cv2.destroyAllWindows()
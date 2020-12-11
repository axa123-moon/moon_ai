import cv2

src = cv2.imread("Image/fruits.jpg", cv2.IMREAD_COLOR)

dst = cv2.resize(src, dsize=(640, 480), interpolation=cv2.INTER_AREA)   # 이미지 재조정 픽셀
dst2 = cv2.resize(src, dsize=(0, 0), fx=0.3, fy=0.7, interpolation=cv2.INTER_LINEAR)    # 이미지 재조정 비율
# cv2.INTER_NEAREST 가장 가까운 이웃 보간법 
# cv2.INTER_LINEAR  쌍 선형 보간법 
# cv2.INTER_LINEAR_EXACT  쌍 선형 보간법 
# cv2.INTER_AREA 영역 보간법 
# cv2.INTER_CUBIC 4×4 바이 큐빅 보간법 
# cv2.INTER_LANCZOS4 8×8 란초스 보간법 

cv2.imshow("src", src)
cv2.imshow("dst", dst)
cv2.imshow("dst2", dst2)
cv2.waitKey(0)
cv2.destroyAllWindows()
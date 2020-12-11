import cv2

src = cv2.imread("Image/cat.jpg")
dst = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)  # 다중 채널 색상 이미지(HSV)로 변환

cv2.imshow("dst", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

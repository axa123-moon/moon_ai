import cv2

src = cv2.imread("Image/fruits.jpg", cv2.IMREAD_COLOR)

dst = src.copy() 
dst2 = src.copy()
roi = src[100:600, 200:700] # 이미지의 특정 영역만 가져옴 x좌표 : 넓이, y좌표 : 높이
dst2[0:500, 0:500] = roi

cv2.imshow("src", src)
cv2.imshow("roi", roi)
cv2.imshow("dst2", dst2)
cv2.waitKey(0)
cv2.destroyAllWindows()
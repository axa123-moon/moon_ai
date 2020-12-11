import cv2


image = cv2.imread("Image/cat.jpg", cv2.IMREAD_ANYCOLOR)

#print(image.shape) # 이미지 정보 조회

cv2.namedWindow("src", flags=cv2.WINDOW_FREERATIO)
cv2.resizeWindow("src", 1920, 1280)
cv2.imshow("src", image)
cv2.waitKey(0)
cv2.destroyWindow("src")
#cv2.destroyAllWindows()
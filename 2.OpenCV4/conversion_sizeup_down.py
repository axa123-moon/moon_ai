import cv2

src = cv2.imread("Image/fruits.jpg", cv2.IMREAD_COLOR)

height, width, channel = src.shape
dst = cv2.pyrUp(src, dstsize=(width*2, height*2), borderType=cv2.BORDER_DEFAULT);   # 이미지 2배 확대
dst2 = cv2.pyrDown(src);    # 이미지 반으로 축소

cv2.imshow("src", src)
cv2.imshow("dst", dst)
cv2.imshow("dst2", dst2)
cv2.waitKey(0)
cv2.destroyAllWindows()
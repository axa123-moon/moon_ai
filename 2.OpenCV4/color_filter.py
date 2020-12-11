import cv2

src = cv2.imread("Image/fire.jpg", cv2.IMREAD_COLOR)

dst = cv2.bilateralFilter(src, 100, 33, 11, borderType=cv2.BORDER_ISOLATED)

dst2 = cv2.boxFilter(src, -1, (5, 5))

cv2.imshow("dst", dst)
cv2.imshow("dst2", dst2)
cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2

src = cv2.imread("Image/cup.jpg", cv2.IMREAD_COLOR)
dst = cv2.flip(src, 0)      # flipCode = 0, 상하 대칭 (X축)
dst1 = cv2.flip(src, 1)     # flipCode > 0, 좌우 대칭 (Y축)
dst2 = cv2.flip(src, -1)    # flipCode < 0, XY축 대칭

cv2.imshow("src", src)
cv2.imshow("dst", dst)
cv2.imshow("dst1", dst1)
cv2.imshow("dst2", dst2)
cv2.waitKey(0)
cv2.destroyAllWindows()
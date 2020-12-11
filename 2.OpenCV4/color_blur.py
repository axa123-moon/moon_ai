import cv2

src = cv2.imread("Image/fire.jpg", cv2.IMREAD_COLOR)

oblur = cv2.blur(src, (9, 9), anchor=(-1, -1), borderType=cv2.BORDER_DEFAULT)
omedianBlur = cv2.medianBlur(src, 9)
oGaussianBlur = cv2.GaussianBlur(src, (9, 9), 1, 1, borderType=cv2.BORDER_DEFAULT)
 
# cv2.blur(원본 이미지, (커널 x크기, 커널 y크기), 앵커 포인트, 픽셀 외삽법)를 이용하여 흐림 효과를 적용합니다.
# 커널 크기는 이미지에 흐림 효과를 적용할 크기를 설정합니다. 크기가 클수록 더 많이 흐려집니다.
# 앵커 포인트는 커널에서의 중심점을 의미합니다. (-1, -1)로 사용할 경우, 자동적으로 커널의 중심점으로 할당합니다.
# 픽셀 외삽법은 이미지를 흐림 효과 처리할 경우, 영역 밖의 픽셀은 추정해서 값을 할당해야합니다.
# 이미지 밖의 픽셀을 외삽하는데 사용되는 테두리 모드입니다. 외삽 방식을 설정합니다.

# cv2.BORDER_CONSTANT iiiiii | abcdefgh | iiiiiii 
# cv2.BORDER_REPLICATE aaaaaa | abcdefgh | hhhhhhh 
# cv2.BORDER_REFLECT fedcba | abcdefgh | hgfedcb 
# cv2.BORDER_WRAP cdefgh | abcdefgh | abcdefg 
# cv2.BORDER_REFLECT_101 gfedcb | abcdefgh | gfedcba 
# cv2.BORDER_REFLECT101 gfedcb | abcdefgh | gfedcba 
# cv2.BORDER_DEFAULT gfedcb | abcdefgh | gfedcba 
# cv2.BORDER_TRANSPARENT uvwxyz | abcdefgh | ijklmno 
# cv2.BORDER_ISOLATED 관심 영역 (ROI) 밖은 고려하지 않음 



cv2.imshow("oblur", oblur)
cv2.imshow("omedianBlur",omedianBlur)
cv2.imshow("oGaussianBlur", oGaussianBlur)
cv2.waitKey(0)
cv2.destroyAllWindows()
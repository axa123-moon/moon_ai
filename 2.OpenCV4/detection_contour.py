import cv2

src = cv2.imread("Image/contour.png", cv2.IMREAD_COLOR)

gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
binary = cv2.bitwise_not(binary)
# 윤곽선(컨투어)를 검출하는 주된 요소는 하얀색의 객체를 검출합니다.
# 그러므로 배경은 검은색이며 검출하려는 물체는 하얀색의 성질을 띄게끔 변형합니다.
# 이진화 처리 후, 반전시켜 검출하려는 물체를 하얀색의 성질을 띄도록 변환합니다.


contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
# cv2.findContours()를 이용하여 이진화 이미지에서 윤곽선(컨투어)를 검색합니다.
# cv2.findContours(이진화 이미지, 검색 방법, 근사화 방법)을 의미합니다.
# 반환값으로 윤곽선, 계층 구조를 반환합니다.
# 윤곽선은 Numpy 구조의 배열로 검출된 윤곽선의 지점들이 담겨있습니다.
# 계층 구조는 윤곽선의 계층 구조를 의미합니다. 각 윤곽선에 해당하는 속성 정보들이 담겨있습니다.

# 검색 방법
# cv2.RETR_EXTERNAL : 외곽 윤곽선만 검출하며, 계층 구조를 구성하지 않습니다.
# cv2.RETR_LIST : 모든 윤곽선을 검출하며, 계층 구조를 구성하지 않습니다.
# cv2.RETR_CCOMP : 모든 윤곽선을 검출하며, 계층 구조는 2단계로 구성합니다.
# cv2.RETR_TREE : 모든 윤곽선을 검출하며, 계층 구조를 모두 형성합니다. (Tree 구조)


# 근사화 방법
# cv2.CHAIN_APPROX_NONE : 윤곽점들의 모든 점을 반환합니다.
# cv2.CHAIN_APPROX_SIMPLE : 윤곽점들 단순화 수평, 수직 및 대각선 요소를 압축하고 끝점만 남겨 둡니다.
# cv2.CHAIN_APPROX_TC89_L1 : 프리먼 체인 코드에서의 윤곽선으로 적용합니다.
# cv2.CHAIN_APPROX_TC89_KCOS : 프리먼 체인 코드에서의 윤곽선으로 적용합니다.


for i in range(len(contours)):
    cv2.drawContours(src, [contours[i]], 0, (0, 0, 255), 1)
    cv2.putText(src, str(i), tuple(contours[i][0][0]), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1)
    print(i, hierarchy[0][i])
    cv2.imshow("src", src)
    cv2.waitKey(0)

cv2.destroyAllWindows()
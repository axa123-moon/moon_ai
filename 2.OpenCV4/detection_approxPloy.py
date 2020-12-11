import cv2

src = cv2.imread("Image/phone.jpg", cv2.IMREAD_COLOR)

gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
binary = cv2.bitwise_not(binary)

contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)

for contour in contours:
    epsilon = cv2.arcLength(contour, True) * 0.02
    # 반복문을 사용하여 색인값과 하위 윤곽선 정보로 반복합니다.
    # 근사치 정확도를 계산하기 위해 윤곽선 전체 길이의 2%로 활용합니다
    # 윤곽선의 전체 길이를 계산하기 위해 cv2.arcLength()을 이용해 검출된 윤곽선의 전체 길이를 계산합니다.
    # cv2.arcLength(윤곽선, 폐곡선)을 의미합니다.
    # 윤곽선은 검출된 윤곽선들이 저장된 Numpy 배열입니다.
    # 폐곡선은 검출된 윤곽선이 닫혀있는지, 열려있는지 설정합니다.
    # •Tip : 폐곡선을 True로 사용할 경우, 윤곽선이 닫혀 최종 길이가 더 길어집니다. (끝점 연결 여부를 의미)
    
    approx_poly = cv2.approxPolyDP(contour, epsilon, True)
    # cv2.approxPolyDP()를 활용해 윤곽선들의 윤곽점들로 근사해 근사 다각형으로 반환합니다.
    # cv2.approxPolyDP(윤곽선, 근사치 정확도, 폐곡선)을 의미합니다.
    # 윤곽선은 검출된 윤곽선들이 저장된 Numpy 배열입니다.
    # 근사치 정확도는 입력된 다각형(윤곽선)과 반환될 근사화된 다각형 사이의 최대 편차 간격을 의미합니다.
    # 폐곡선은 검출된 윤곽선이 닫혀있는지, 열려있는지 설정합니다.
    # •Tip : 근사치 정확도의 값이 낮을 수록, 근사를 더 적게해 원본 윤곽과 유사해집니다.

    for approx in approx_poly:
        cv2.circle(src, tuple(approx[0]), 3, (255, 0, 0), -1)
        # 다시 반복문을 통해 근사 다각형을 반복해 근사점을 이미지 위에 표시합니다.

src2 = cv2.resize(src, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR) 

cv2.imshow("src", src2)
cv2.waitKey(0)
cv2.destroyAllWindows()
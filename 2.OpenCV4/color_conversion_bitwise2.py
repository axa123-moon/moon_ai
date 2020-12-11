import cv2

src = cv2.imread("Image/tomato.jpg", cv2.IMREAD_COLOR)
hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)  # 다중 채널 색상 이미지(HSV)로 변환
h, s, v = cv2.split(hsv)                    # 색상을 3개 채널로 분리

h_red = cv2.inRange(h, 0, 5)    # 빨간색 범위 색상 데이타만 추출
dst = cv2.bitwise_and(hsv, hsv, mask = h_red)   # h_red 영역만 보이도록 mask 처리함
dst = cv2.cvtColor(dst, cv2.COLOR_HSV2BGR)      # HSV 색상공간을 BGR 색상공간으로 변환
                                                # imshow()함수는 BGR 색상공간만 정상적으로 출력하기 때문

cv2.imshow("orange", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2

src = cv2.imread("Image/tomato.jpg", cv2.IMREAD_COLOR)
hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)  # 다중 채널 색상 이미지(HSV)로 변환
h, s, v = cv2.split(hsv)                    # 색상을 3개 채널로 분리

orange = cv2.inRange(hsv, (8, 100, 100), (20, 255, 255))    # 오랜지색 범위 색상 데이타만 추출
blue = cv2.inRange(hsv, (110, 100, 100), (130, 255, 255))   # 파란색 범위 색상 데이타만 추출
mix_color = cv2.addWeighted(orange, 1.0, blue, 1.0, 0.0)    # 가중치 없이 병합
                                                            # alpha = 1.0, beta = 1.0, gamma = 0.0

dst = cv2.bitwise_and(hsv, hsv, mask = mix_color)   # h_red 영역만 보이도록 mask 처리함
dst = cv2.cvtColor(dst, cv2.COLOR_HSV2BGR)          # HSV 색상공간을 BGR 색상공간으로 변환
                                                    # imshow()함수는 BGR 색상공간만 정상적으로 출력하기 때문

cv2.imshow("orange", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
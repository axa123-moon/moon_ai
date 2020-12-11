import numpy as np
import cv2

src = np.zeros((768, 1366, 3), dtype = np.uint8)    # 이미지 배열 생성 

cv2.line(src, (100, 100), (1200, 100), (0, 0, 255), 3, cv2.LINE_AA)     # 선 그리기
cv2.circle(src, (300, 300), 50, (0, 255, 0), cv2.FILLED, cv2.LINE_4)    # 원 그리기
cv2.rectangle(src, (500, 200), (1000, 400), (255, 0, 0), 5, cv2.LINE_8) # 사각형 그리기
cv2.ellipse(src, (1200, 300), (100, 50), 0, 90, 180, (255, 255, 0), 2)  # 호 그리기

pts1 = np.array([[100, 500], [300, 500], [200, 600]])
pts2 = np.array([[600, 500], [800, 500], [700, 600]])
pts3 = np.array([[600, 500], [800, 500], [700, 600], [700, 650]])
cv2.polylines(src, [pts1, pts3], True, (0, 255, 255), 2)                      # 다각형 그리기
cv2.fillPoly(src, [pts2], (255, 0, 255), cv2.LINE_AA)                   # 다각형 채우기

cv2.putText(src, "HELLO! JHON", (900, 600), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 3, bottomLeftOrigin=False )  # 글자 그리기

cv2.imshow("src", src)
cv2.waitKey(0)
cv2.destroyAllWindows()
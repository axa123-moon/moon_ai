import numpy as np
import cv2

src = cv2.imread('Image/zebra.jpg')

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (9, 9))
# cv2.getStructuringElement()를 활용해 구조요소를 생성합니다.
# cv2.getStructuringElement(커널의 형태, 커널의 크기, 중심점)로 구조 요소을 생성합니다.
# 커널의 형태는 직사각형(Rect), 십자가(Cross), 타원(Ellipse)이 있습니다.
# 커널의 크기는 구조 요소의 크기를 의미합니다. 이 때, 커널의 크기가 너무 작다면 커널의 형태는 영향을 받지 않습니다.
# 고정점은 커널의 중심 위치를 나타냅니다. 필수 매개변수가 아니며, 설정하지 않을 경우 사용되는 함수에서 값이 결정됩니다.

dilate = cv2.dilate(src, kernel, anchor=(-1, -1), iterations=5)
erode = cv2.erode(src, kernel, anchor=(-1, -1), iterations=5)
# cv2.dilate(원본 배열, 구조 요소, 고정점, 반복 횟수, 테두리 외삽법, 테두리 색상)로 팽창 연산을 진행합니다.
# cv2.erode(원본 배열, 구조 요소, 고정점, 반복 횟수, 테두리 외삽법, 테두리 색상)로 침식 연산을 진행합니다.
# 단, 팽창 연산의 경우 밝은 영역이 커지며, 침식 연산의 경우 어두운 영역이 커집니다.

dst = np.concatenate((src, dilate, erode), axis=1)

dst = cv2.resize(dst, dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)    # 이미지 크기 축소
cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
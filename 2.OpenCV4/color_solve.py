import cv2
import numpy as np

src1 = np.array([[9, 2], [1, 1]], dtype=np.double)
src2 = np.array([38, 5], dtype=np.double)

dst = cv2.solve(src1, src2, flags=cv2.DECOMP_LU)    # 가우스 소거법을 이요하여 해를 찾음
                                                    # 해를 찾을 경우 True와 값을 반환하며, 못찾을 경우 False 반환함

print(dst)
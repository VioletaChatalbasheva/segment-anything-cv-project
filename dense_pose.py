import os
import cv2
import numpy as np

img = cv2.imread(os.path.join(os.getcwd(), 'segment_anything', 'lower_body', 'dense', '013563_5.png'))
print(np.unique(img))

img_color = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
print(np.unique(img_color))

cv2.imwrite(os.path.join(os.getcwd(), 'segment_anything', 'lower_body', 'new_dense', '013563_5.png'), img_color)
# cv2.imshow("result", img_color)
# cv2.waitKey(0)

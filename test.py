import cv2 as cv
import time
img = cv.imread("/Users/jakehopkins/Downloads/16-18-58-775_pred_0.0003.jpg")
cv.imshow("pre", img)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.imshow("after", img)
cv.waitKey(0)              # blocks until any keypress, keeps windows alive
cv.destroyAllWindows()
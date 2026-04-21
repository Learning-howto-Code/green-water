import os
import cv2 as cv
import numpy as np

old = "/Users/jakehopkins/Downloads/if_water/poop/test/poop/, 2026-04-11-19-13-51-122.jpg"
new_path = "/Users/jakehopkins/Downloads/if_water/poop/test/poop/, 2026-04-11-19-13-51-621.jpg"
out_dir = "/Users/jakehopkins/Downloads/if_water/poop/diff/"
old = cv.imread(old)
new = cv.imread(new_path)

old = cv.resize(old, (224, 224))
new = cv.resize(new, (224, 224))

new_grey = cv.cvtColor(new, cv.COLOR_BGR2GRAY)


diff = cv.absdiff(old, new)
grey_diff = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)


combo = np.concatenate([new_grey, grey_diff], axis=-1)  # shape (1, 224, 224, 6)

# cv.imshow("new", new)
# cv.imshow("diff", diff)
cv.imshow("combo", combo)

cv.waitKey(0)
cv.destroyAllWindows()

base = os.path.basename(new_path)
out_path = os.path.join(out_dir, f"{base}_diff.jpg")
cv.imwrite(out_path, diff)
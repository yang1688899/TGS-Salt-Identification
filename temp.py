import utils

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2


train_df,test_df = utils.generate_dataframe()
img_paths = list(train_df["img_path"])
mask_paths = list(train_df["mark_path"])

for path in img_paths:
    img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    cv2.imshow("origin",img)
    cv2.waitKey(0)
    flip_img = utils.random_filp(img)
    cv2.imshow("flip",flip_img)
    cv2.waitKey(0)

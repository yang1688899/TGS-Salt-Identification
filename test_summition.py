from tqdm import tqdm_notebook
import numpy as np
from keras.models import load_model
from matplotlib.pyplot import plot as plt
import pandas as pd

import utils

thresholds = np.linspace(0, 1, 50)
# ious = np.array([iou_metric_batch(y_valid_ori, np.int32(preds_valid > threshold)) for threshold in tqdm_notebook(thresholds)])

save_obj = utils.load_pickle("./save_obj.p")

val_features = np.array(save_obj["features"])
val_labels = np.array(save_obj["labels"])
test_df = save_obj["test_df"]

model = load_model("./keras.model")

val_predict = model.predict(val_features)

val_predict = [utils.downsample(x) for x in val_predict]
val_labels = [utils.downsample(x) for x in val_labels]

thresholds = np.linspace(0, 1, 50)
ious = np.array([utils.iou_metric_batch(val_labels, np.int32(val_predict > threshold)) for threshold in tqdm_notebook(thresholds)])
threshold_best_index = np.argmax(ious[9:-10]) + 9
iou_best = ious[threshold_best_index]
threshold_best = thresholds[threshold_best_index]

#plot the thresholds_ious comparation
plt.plot(thresholds, ious)
plt.plot(threshold_best, iou_best, "xr", label="Best threshold")
plt.xlabel("Threshold")
plt.ylabel("IoU")
plt.title("Threshold vs IoU ({}, {})".format(threshold_best, iou_best))
plt.legend()

test_img_paths = list(test_df["img_path"])
test_features = np.array([utils.load_feature(path) for path in test_img_paths])
test_preds = model.predict(test_features)

pred_dict = {idx: utils.RLenc(np.round(utils.downsample(test_preds[i]) > threshold_best)) for i, idx in enumerate(tqdm_notebook(test_df.index.values))}
sub = pd.DataFrame.from_dict(pred_dict,orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv('submission.csv')
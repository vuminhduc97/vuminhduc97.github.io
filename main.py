import cv2
import numpy as np
import time
import sys
import os

from predict_detect import TextDetector
from log import get_logger
from data import create_operators, transform
import util as utility
from postprocess.build_post_process import build_post_process
from get_file import get_image_file_list
from process_box import sorted_boxes, get_rotate_crop_image
from predict_cls import TextClassifier


logger = get_logger()
args = utility.parse_args()
image_file_list = get_image_file_list(args.image_dir)
text_detector = TextDetector(args)
count = 0
total_time = 0
draw_img_save = "./inference_results"
for image_file in image_file_list:
    img = cv2.imread(image_file)
    dt_boxes, elapse = text_detector(img)
    if count > 0:
        total_time += elapse
    count += 1
    logger.info('predict time of {}: {}'.format(image_file, elapse))
    src_im = utility.draw_text_det_res(dt_boxes, image_file)
    img_name_pure = os.path.split(image_file)[-1]
    img_path = os.path.join(draw_img_save,
                            "det_res_{}".format(img_name_pure))
    cv2.imwrite(img_path, src_im)
    logger.info("The visualized image saved in {}".format(img_path))
if count > 1:
    logger.info("Avg Time: {}".format(total_time / (count - 1)))


import cv2
import numpy as np
from blurgenerator import motion_blur_with_depth_map

def depth_map_overlay(image_path, depth_map_path):

    # Загрузка изображений
    img = cv2.imread(image_path)
    depth_img = cv2.imread(depth_map_path)

    depth_img = cv2.resize(depth_img, (img.shape[1], img.shape[0]))

    result = motion_blur_with_depth_map(
    img,
    depth_map=depth_img,
    angle=30,
    num_layers=10,
    min_blur=1,
    max_blur=50
    )


    cv2.imshow('Overlay', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('./output.png', result)

depth_map_overlay('depth-test.jpg', 'depth-map-test.png')
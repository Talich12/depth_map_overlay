import cv2
import numpy as np

def depth_map_overlay(image_path, depth_map_path):

    # Загрузка изображений
    img = cv2.imread(image_path)

    # Загрузите изображение в оттенках серого
    depth_img = cv2.imread(depth_map_path , cv2.IMREAD_GRAYSCALE)

    # Преобразуйте изображение в цветовую карту
    colored_depth_img = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)

    # Сохраните преобразованное изображение
    cv2.imwrite('colored_depth_map.png', colored_depth_img)
    alpha = 0.5

    # Наложите изображения
    overlay = cv2.addWeighted(img, alpha, colored_depth_img, 1 - alpha, 0)

    #cv2.imshow('Overlay', overlay)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    cv2.imwrite('./output.png', overlay)

depth_map_overlay('depth-test.jpg', 'depth-map-test.png')
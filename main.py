import cv2
import numpy as np


def depth_map_overlay(image_path, depth_map_path):

    if depth_map_path.endwith('.json'):
        with open(depth_map_path, 'r') as f:
            data = json.load(f)

        # Преобразование списка в numpy массив
        depth_map = np.array(data)

        # Нормализация значений глубины до диапазона 0-255
        normalized_depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Преобразование нормализованного массива глубины в изображение
        depth_image = cv2.imencode('.jpg', normalized_depth_map)[1]

    elif depth_map_path.endswith('.jpeg') or depth_map_path.endswith('.png'):
        depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)

    # Загрузка изображений
    image = cv2.imread(image_path)
    # Приведение карты глубины к тому же размеру, что и изображение
    depth_map = cv2.resize(depth_map, (image.shape[1], image.shape[0]))

    # Наложение карты глубины на изображение
    overlay = cv2.addWeighted(image, 0.7, depth_map, 0.3, 0)

    cv2.imshow('Overlay', overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
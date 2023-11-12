import os
import shutil
import csv
import cv2
import numpy as np
from ultralytics import YOLO
from model import predict_model
from shapely.geometry import Polygon, box

PATH_TO_DANGERS_ZONES = './sorted_train_dataset/danger_zones'
PATH_TO_MODEL = 'model/trained_model.pt'


def draw_human_on_image(path_to_image, coordinates_zona, destination_path):
    # Загрузка фотографии
    image = cv2.imread(path_to_image)

    pts = np.array(coordinates_zona, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(image, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

    # Сохраняем измененную фотографию
    cv2.imwrite(destination_path, image)

    # Отображаем фотографию с нарисованным прямоугольником (для просмотра)
    cv2.imshow('Rectangle', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_percent_intersection(coordinates, zones):
    maxi_result = 0
    for current_zone in zones:
        # Координаты прямоугольника (верхний левый угол и нижний правый угол)
        rect_coords = coordinates
        # Координаты многоугольника
        poly_coords = current_zone

        # Создание прямоугольника и многоугольника
        rectangle = box(*rect_coords)
        polygon = Polygon(poly_coords)

        # Вычисление пересечения
        intersection = rectangle.intersection(polygon)

        # Вычисление площадей
        area_rectangle = rectangle.area
        area_intersection = intersection.area

        # Процент пересечения
        percent_intersection = (area_intersection / area_rectangle) * 100

        maxi_result = max(maxi_result, percent_intersection)

    return maxi_result


def get_danger_zones(path_to_data_set):
    result = {}
    for file_name in os.listdir(path_to_data_set):
        if file_name.endswith('.txt'):
            text = open(path_to_data_set + '/' + file_name).read()

            name_zona = file_name.split('danger_')[1][:-4].split('_')[0]
            if not (name_zona in result.keys()):
                result[name_zona] = []

            cleaned_text = text.replace("[", "").replace('\n', ' ')
            pairs = cleaned_text.split("], ")

            # Преобразовываем строки в целые числа и формируем массив
            result[name_zona].append([list(map(int, pair.replace(']', '').split(', '))) for pair in pairs])
    return result


def analysis_folder():
    print('Загружаются координаты опасных зон... [1/1]')
    zones = get_danger_zones(PATH_TO_DANGERS_ZONES)
    print('Загружается обученная модель... [1/1]')
    model = YOLO(PATH_TO_MODEL)
    print('-------------------')

    path_to_folder = input('Введите путь к папке с камерами: ')
    print('-------------------')

    with open('result.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        field = ["camera_name", "frame_filename", "in_danger_zone", 'percent']
        writer.writerow(field)

        i = 0
        for camera_name in os.listdir(path_to_folder):
            print(f'Прогесс [{i+1}/{len(os.listdir(path_to_folder))}]')
            selected_zona = zones[camera_name]
            for filename in os.listdir(os.path.join(path_to_folder, camera_name)):
                coordinates = predict_model.get_coordinates_humans(model,
                                                                   os.path.join(path_to_folder, camera_name, filename))

                for human in coordinates:
                    percent_intersection = get_percent_intersection(human, selected_zona)
                    if percent_intersection >= 15:
                        res = 'True'
                    else:
                        res = 'False'
                    writer.writerow([camera_name, filename, res, str(percent_intersection / 100)])
            i += 1


def analysis_one_photo():
    print('Загружаются координаты опасных зон... [1/1]')
    zones = get_danger_zones(PATH_TO_DANGERS_ZONES)
    print('Загружается обученная модель... [1/1]')
    model = YOLO(PATH_TO_MODEL)
    print('-------------------')

    print('Номер | Имя камеры')
    i = 0
    for name in list(zones.keys()):
        print(f'{i}.\t{name}')
        i += 1
    id_zone = int(input('Выберите номер камеры: '))
    print('-------------------')

    path_to_image = input('Введите путь фото для ее распознавания: ')
    print('-------------------')

    print('Модель распознает человека...')
    selected_zona = zones[list(zones.keys())[id_zone]]
    coordinates = predict_model.get_coordinates_humans(model, path_to_image)
    print('-------------------')

    print('Высчитываем пересечение человека с опасной зоной...')
    percent_intersection = get_percent_intersection(coordinates, selected_zona)
    print('-------------------')

    print('Результат: ', end='')
    if percent_intersection >= 15:
        print(f'true ({percent_intersection})')
    else:
        print(f'false ({percent_intersection})')
    print('-------------------')

    draw_human_on_image(os.path.join('./runs/detect/predict/', path_to_image.split('/')[-1]),
                        zones[list(zones.keys())[id_zone]],
                        './result.jpg')

    shutil.rmtree('./runs/detect/predict/')


def main():
    mode = input('Выберите режим: 0 - проанализировать тестовые данные, 1 - проанализировать конкретную фотографию: ')
    if mode == '0':
        analysis_folder()
    elif mode == '1':
        analysis_one_photo()


if __name__ == '__main__':
    main()

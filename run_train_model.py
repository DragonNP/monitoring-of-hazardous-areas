import os
import random
import shutil

import cv2
import numpy as np

import uuid

from model import train_model


def clear_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        # Перебираем все элементы в папке
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                # Если это файл, удаляем его
                if os.path.isfile(file_path):
                    os.remove(file_path)
                # Если это папка, удаляем ее и все ее содержимое
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Ошибка при удалении {file_path}: {e}")


def distort_image(path_to_image, destination_folder, output_file_name=''):
    # Загрузка изображения
    original_image = cv2.imread(path_to_image)

    # Изменение яркости и контраста
    alpha = 1.0 + random.uniform(-0.5, 0.5)  # Изменение контраста
    beta = random.randint(-50, 50)  # Изменение яркости
    distorted_image = cv2.convertScaleAbs(original_image, alpha=alpha, beta=beta)

    # Добавление случайного шума
    noise = np.random.normal(0, 0.63, original_image.shape).astype(np.uint8)
    distorted_image = cv2.add(distorted_image, noise)

    # Сохранение исходного и искаженного изображений
    if output_file_name == '':
        output_file_name = path_to_image.split('/')[-1]
    cv2.imwrite(os.path.join(destination_folder, output_file_name), distorted_image)


def add_distort_to_dataset(path_to_dataset: str, count: int):
    print('Добавляются шумы к дата сету.. [1/1]')
    for img_type in ['predict', 'train', 'val']:
        path_images = os.path.join(path_to_dataset, 'images', img_type)
        path_txt = os.path.join(path_to_dataset, 'labels', img_type)
        for img_name in random.sample(os.listdir(path_images), count):
            new_name = str(uuid.uuid4())
            path_to_image = os.path.join(path_images, img_name)
            path_to_img_txt = os.path.join(path_txt, '.'.join(img_name.split('.')[:-1]) + '.txt')
            distort_image(path_to_image, os.path.join(path_to_dataset, 'images', img_type),
                          new_name + '.' + img_name.split('.')[-1])
            shutil.copy2(path_to_img_txt, os.path.join(path_txt, new_name + '.txt'))
    print('-' * 20)


def sort_dataset(path_to_dataset: str, path_to_sorted_dataset: str = ''):
    print('Датасет сортируется... [1/2]')

    if path_to_sorted_dataset == '':
        path = '/'.join(path_to_dataset.split('/')[0:-1])
        new_name_folder = 'sorted_' + path_to_dataset.split('/')[-1]
        path_to_sorted_dataset = os.path.join(path, new_name_folder)
    clear_folder(path_to_sorted_dataset)

    destination_submain_folder = os.path.join(path_to_sorted_dataset, 'unsorted')
    destination_folder1 = os.path.join(destination_submain_folder, 'train')
    destination_folder2 = os.path.join(destination_submain_folder, 'val')
    destination_folder3 = os.path.join(destination_submain_folder, 'predict')

    clear_folder(path_to_sorted_dataset)
    clear_folder(destination_submain_folder)
    clear_folder(destination_folder1)
    clear_folder(destination_folder2)
    clear_folder(destination_folder3)

    probabilities = [0.86, 0.10, 0.04]
    destination_folders = [destination_folder1, destination_folder2, destination_folder3]

    for camera_name in os.listdir(path_to_dataset):
        source_folder = os.path.join(path_to_dataset, camera_name)
        all_files = os.listdir(source_folder)

        for i in range(0, len(all_files), 2):
            file_name = all_files[i]
            source_path_image = os.path.join(source_folder, file_name)
            source_path_data = os.path.join(source_folder, '.'.join(file_name.split('.')[:-1]) + '.txt')

            # Генерируем случайное число
            random_number = random.random()

            # Находим папку в соответствии с вероятностью
            cumulative_probability = 0
            for folder, probability in zip(destination_folders, probabilities):
                cumulative_probability += probability
                if random_number <= cumulative_probability:
                    new_name = str(uuid.uuid4())

                    shutil.copy2(source_path_image, os.path.join(folder, new_name + '.' + file_name.split('.')[-1]))
                    shutil.copy2(source_path_data, os.path.join(folder, new_name + '.txt'))
                    break

    print('Датасет сортируется... [2/2]')
    destination_submain2_folder = os.path.join(path_to_sorted_dataset, 'sorted')
    clear_folder(destination_submain2_folder)
    clear_folder(os.path.join(destination_submain2_folder, 'images'))
    clear_folder(os.path.join(destination_submain2_folder, 'labels'))

    for folder_name in os.listdir(destination_submain_folder):
        clear_folder(os.path.join(destination_submain2_folder, 'images', folder_name))
        clear_folder(os.path.join(destination_submain2_folder, 'labels', folder_name))
        for file_name in os.listdir(os.path.join(destination_submain_folder, folder_name)):
            if file_name[-3::] == 'jpg':
                shutil.move(os.path.join(destination_submain_folder, folder_name, file_name),
                            os.path.join(destination_submain2_folder, 'images', folder_name, file_name))
            else:
                shutil.move(os.path.join(destination_submain_folder, folder_name, file_name),
                            os.path.join(destination_submain2_folder, 'labels', folder_name, file_name))

    shutil.move(os.path.join(destination_submain2_folder, 'images'), path_to_sorted_dataset)
    shutil.move(os.path.join(destination_submain2_folder, 'labels'), path_to_sorted_dataset)
    shutil.rmtree(destination_submain_folder)
    shutil.rmtree(destination_submain2_folder)


    print('-' * 20)


def main():
    print('-' * 20)
    path_to_dataset = input('Введите путь к датасету: ')
    is_dataset_nosorted = input('Отсортировать модель? (1 - да, 0 - нет): ')
    add_distort = input('Наложить шумы к датасету? (1 - да, 0 - нет): ')
    print('-' * 20)

    if is_dataset_nosorted == '1':
        sort_dataset(path_to_dataset,
                     './sorted_train_dataset')

    if add_distort == '1':
        print('Добавляются шумы к фотографиям датасета... [1/]')
        add_distort_to_dataset('./sorted_train_dataset', 25)
        print('-' * 20)

    print('Начинается процесс обучения.. ')
    train_model.train_model()
    print('Обучение завершено')
    print('-' * 20)


if __name__ == '__main__':
    main()

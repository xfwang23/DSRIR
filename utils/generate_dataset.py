import os
import csv


def generate_noise_BSD400():
    image_dir = r'/home/sht/all_in_one_dataset/BSD400'
    imgs = os.listdir(image_dir)
    with open('../datasets/train/train_noise.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for img in imgs:
            image_path = os.path.join('BSD400/', img)

            writer.writerow([image_path, image_path])
        f.close()


def generate_noise_WED():
    image_dir = r'/home/sht/all_in_one_dataset/WED/gt'
    imgs = os.listdir(image_dir)
    with open('../datasets/train/train_noise.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        for img in imgs:
            image_path = os.path.join('WED/gt/', img)

            writer.writerow([image_path, image_path])
        f.close()


def generate_rain_Rain100L():
    image_dir = r'/home/sht/all_in_one_dataset/Rain100L/train/rainy'
    imgs = os.listdir(image_dir)
    with open('../datasets/train/train_rain.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for img in imgs:
            rainy_path = os.path.join('Rain100L/train/rainy', img)
            clean_path = os.path.join('Rain100L/train/gt', 'no' + img)
            writer.writerow([clean_path, rainy_path])
        f.close()


def generate_hazy_Reside():
    image_dir = r'/home/sht/all_in_one_dataset/Reside/train/hazy'
    imgs = os.listdir(image_dir)
    with open('../datasets/train/train_haze.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for img in imgs:
            rainy_path = os.path.join('Reside/train/hazy', img)
            clean_path = os.path.join('Reside/train/gt', img.split('_')[0] + '.jpg')
            writer.writerow([clean_path, rainy_path])
        f.close()


if __name__ == '__main__':
    generate_hazy_Reside()

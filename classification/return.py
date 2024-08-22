import os
from PIL import Image
import shutil

# 定义路径
txt_file_path = 'val_set.txt'  # 存储图片路径的txt文件
# Input_folder = 'playground/experiments/Imagenet_condi1'  # 保存图片的文件夹

Input_folder = '../Clip_codec/playground/experiments/Imagenet_condi5'  # 保存图片的文件夹

# 读取txt文件中的路径
with open(txt_file_path, 'r') as file:
    paths = file.readlines()


# 处理每个图片路径
for idx, path in enumerate(paths):
    path = path.strip()  # 去掉多余的空白字符
    img_name = path.split('/')[-1]
    image_name = img_name.split('.')[0]
    # 创建输出文件夹（如果不存在的话）
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    shutil.copy(os.path.join(Input_folder, f'{image_name}.png'), os.path.join(directory, f'{img_name}.png'))
    print(f'保存图片到 {path}')


print('所有图片处理完毕！')

import PIL.Image as Image
import os, json
import numpy as np

# 将标签label_data_0313.json中的图片水平翻转
old_img_dir = 'clips'
base_dir = 'D:/lanenet/'
json_file = base_dir + 'label_data_0313.json'
dict = {}
with open(base_dir + 'flip_label_data_0313.json','w',encoding='utf-8') as f:
    for line in open(json_file).readlines():
        info_dict = json.loads(line)
        h_samples = info_dict['h_samples']
        lanes = info_dict['lanes']
        print(lanes)
        old_img_path = info_dict['raw_file']
        full_img_path = base_dir + old_img_path
        img = Image.open(full_img_path)
        flip_img = img.transpose(Image.FLIP_LEFT_RIGHT)  # p表示概率  水平翻转

        flip_img_path = base_dir + old_img_path.replace(old_img_dir, 'flips')

        if not os.path.exists(flip_img_path.replace(flip_img_path.split('/')[-1],'')):
            os.makedirs(flip_img_path.replace(flip_img_path.split('/')[-1],''))
        #保存翻转后的图片
        flip_img.save(flip_img_path)
        img = np.array(img)
        h,w,_ = img.shape
        #水平翻转，所以只改横坐标
        flip_lanes = []
        for i in lanes[0]:
            if int(i) == -2:
                new_x = -2
            else:
                new_x = w - i
            flip_lanes.append(new_x)
        flip_lanes_list = [flip_lanes]
        print(flip_lanes_list)
        dict['lanes'] = flip_lanes_list
        dict['h_samples'] = h_samples
        dict['raw_file'] = flip_img_path
        print(dict)
        f.write(str(dict) + '\n')







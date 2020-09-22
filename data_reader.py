import os
import numpy as np
from PIL import Image, ImageOps
import paddle.fluid as fluid

#定义基础函数
def RandomHorizonFlip(img):
    i = np.random.rand()
    if i > 0.5:
        img = ImageOps.mirror(img)
    return img

def RandomCrop(img, crop_w, crop_h):
    w, h = img.size[0], img.size[1]
    i = np.random.randint(0, w - crop_w)
    j = np.random.randint(0, h - crop_h)
    return img.crop((i, j, i + crop_w, j + crop_h))

def data_reader(list_filename, mode):
    def reader():
        if mode == "TRAIN":
            shuffle = True
        else:
            shuffle = False

        lines = open(list_filename).readlines()
        if shuffle:
            np.random.shuffle(lines)
        
        for line in lines:
            line = line.strip('\n\r\t').split(' ')
            filename = line[0]
            img = Image.open(filename).convert('RGB')
            #对训练数据处理
            if mode == "TRAIN":
                img = RandomHorizonFlip(img)
                img = img.resize((256 + 30, 256 + 30), Image.BILINEAR)
                img = RandomCrop(img, 256, 256)
            else:
                #对测试数据处理
                img = img.resize((256, 256), Image.BILINEAR)

            img = np.array(img).astype('float32')
            img = img.transpose((2, 0, 1))
            img = (img / 255.0 - 0.5) / 0.5
            #yield img, int(lab)
            yield img
    return reader

#测试数据读取器
# trainA_list = os.path.join('/home/aistudio/data', 'trainA.txt')
# trainA_reader = fluid.io.batch(data_reader(trainA_list, mode="TRAIN"), batch_size=1, drop_last=False)

# with fluid.dygraph.guard():
#     for step in range(1, 11):
#         try:
#             real_A = next(trainA_iter)
#         except:
#             trainA_iter = iter(trainA_reader())
#             real_A = next(trainA_iter)
        
#         real_A = fluid.dygraph.to_variable(np.array(real_A))
#         print(real_A.numpy().shape)

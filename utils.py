
import base64
import numpy as np
import cv2

def get_picture_base64_data(image_path):
    with open(image_path, 'rb') as img_obj:
        base64_data = base64.b64encode(img_obj.read())
    return base64_data


#解决cv2.imread不支持中文路径问题
def cv2_readimg(filename, mode):
    #把图片文件存入内存
	img_date= np.fromfile(filename, dtype=np.uint8)  
    #从内存数据读入图片
	img = cv2.imdecode(img_date, mode)  
	return img
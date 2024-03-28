from conf.setting import *

import cv2
import shutil
import json
from tqdm import trange, tqdm
from recognition import FlowchartRecognition
import threading
import torch
from PIL import Image
from FasterRCNN.predict import flowchat_recognize, create_model
from cnocr import CnOcr

def init_flowchart_recognize_model(classed_file= "FasterRCNN/save_weights/V15ArrowMix/classes.json", models_path = "./FasterRCNN/save_weights/V15ArrowMix/resNetFpn-model-15.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # read class_indict
    assert os.path.exists(classed_file), "json file {} dose not exist.".format(classed_file)
    with open(classed_file, 'r') as f:
        class_dict = json.load(f)
    category_index = {str(v): str(k) for k, v in class_dict.items()}
    
    # create model
    recognize_model = create_model(num_classes=len(class_dict)+1)
    # load train weights
    weights_path = models_path
    assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    recognize_model.load_state_dict(weights_dict)
    recognize_model.to(device)
    recognize_model.eval()  # 进入验证模式
    return recognize_model, category_index

def start_recognize(recognize_model, category_index, ocr_model, img_path, result_save_path, img_name, recognize_lock, cnocr_lock, file_create_lock):
    recognizer = FlowchartRecognition(recognize_model, category_index, ocr_model, recognize_lock, cnocr_lock, file_create_lock)
    shape_nodes, arrow_nodes = recognizer.recognize_flowchart(image_path=f"{img_path}/{img_name}", 
                                                                img_name=f"{img_path.split('/')[-1]}/{img_name}")
    tmp_shape_nodes = list()
    for item in shape_nodes:
        tmp_shape_nodes.append({
            "id": item["id"],
            "Name": item["Name"],
            "coordinate": item["coordinate"],
            "top": item["top"],
            "left": item["left"],
            "width": item["size"]["width"],
            "height": item["size"]["height"],
            "rows": item["size"]["rows"]
        })
    tools.write_2_json({"nodes": tmp_shape_nodes, "edges": arrow_nodes}, f"{result_save_path}/{img_name.replace('.png', '.json').replace('.jpg', '.json')}")

def start_recognize_flowchart(flowchart_img_savepath, result_savepath):
    
    # recognizer = FlowchartRecognition()
    recognize_model, category_index = init_flowchart_recognize_model()
    # ocr_model = CnOcr(rec_model_name='densenet_lite_136-gru', rec_model_backend="pytorch", det_model_name="db_resnet34", det_model_backend="pytorch") # OCR中文
    ocr_model = CnOcr(det_model_name='en_PP-OCRv3_det', rec_model_name='en_PP-OCRv3') # OCR英文
    print(f"recognize_model:{recognize_model}")
    print(f"category_index:{category_index}")
    run_paras = list()
    recognize_lock = threading.Lock()
    cnocr_lock = threading.Lock()
    file_create_lock = threading.Lock()
    folders = os.listdir(flowchart_img_savepath)
    for folder in folders:
        flowchart_imgs = os.listdir(f"{flowchart_img_savepath}/{folder}")
        for img_name in tqdm(flowchart_imgs[:], total=len(flowchart_imgs), desc=f"{folder}: "):            
            run_paras.append((recognize_model, category_index, ocr_model, f"{flowchart_img_savepath}/{folder}", f"{result_savepath}/{folder}", f"{img_name}", recognize_lock, cnocr_lock, file_create_lock))
            
    tools.multi_thread_run(32, start_recognize, run_paras, "Recognize Flowchart: ")
            

def save_cover_shape_imgs(flowchart_data_path, flowhcart_img_path, file_name, result_save_path, cover_img_save_path):
    try:
        flowchart_data = tools.read_json(f"{flowchart_data_path}/{file_name.replace('.png', '.json')}")
    except:
        return
    image = cv2.imread(f"{flowhcart_img_path}/{file_name}")
    for item in flowchart_data['nodes']:
        box = item["coordinate"]
        cv2.rectangle(image, (int(box[0]), int(box[2])), (int(box[1]), int(box[3])), (255, 255, 255), -1)
    
    if not os.path.exists(cover_img_save_path): os.makedirs(cover_img_save_path)
    cv2.imwrite(f"{cover_img_save_path}/{file_name}", image)

if __name__=='__main__':
    start_recognize_flowchart(flowchart_img_savepath="images", result_savepath="results")
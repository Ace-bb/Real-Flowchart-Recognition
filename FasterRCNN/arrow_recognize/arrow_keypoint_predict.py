import os, json, cv2, numpy as np, matplotlib.pyplot as plt
import math

import torch
import time

import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision import transforms

from PIL import Image

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def get_model(num_keypoints, weights_path=None):
    
    anchor_generator = AnchorGenerator(sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0))
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=False,
                                                                pretrained_backbone=True,
                                                                num_keypoints=num_keypoints,
                                                                num_classes = 2, # Background is the first class, object is the second class
                                                                rpn_anchor_generator=anchor_generator)

    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)        
        
    return model

def draw_res(img_path, res_shapes, save_path):
    img = cv2.imread(img_path)
    for shape in res_shapes:
        cv2.rectangle(img, (int(shape["boxes"][0]), int(shape["boxes"][1])), (int(shape["boxes"][2]), int(shape["boxes"][3])), (0,0,255), 1)
        cv2.circle(img, (int(shape['start'][0]), int(shape['start'][1])), 3, (255,0,0), 2)
        cv2.circle(img, (int(shape['end'][0]), int(shape['end'][1])), 3, (255,0,0), 2)
        cv2.putText(img, "start angle : {0:0.2f}".format(shape['angle']),
                        (int(shape['start'][0]), int(shape['start'][1])), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 255), 1)
    cv2.imwrite(f"{save_path}/keypoint_arrow.jpg", img)

def angle_beween_points(a, b):
    # arrow_slope = (a[0] - b[0]) / (a[1] - b[1])
    if b[0] - a[0] == 0:
        if b[1] - a[1] > 0:
            arrow_slope = float("inf")
        else:
            arrow_slope = -float("inf")
    else:
        arrow_slope =  (b[1] - a[1]) / (b[0] - a[0])
        
    arrow_angle = math.degrees(math.atan(arrow_slope))
    a = float("inf")
    return arrow_angle

def recognize(arrow_img_path, model_path, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    model = get_model(num_keypoints=2, weights_path=model_path)
    model.to(device)
    
    # load image
    image_path = arrow_img_path
    original_img = Image.open(image_path).convert("RGB")
    
    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    
    model.eval()
    
    with torch.no_grad():
        # init
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)
        
        t_start = time_synchronized()
        predictions = model(img.to(device))[0]
        t_end = time_synchronized()
        print("inference+NMS time: {}".format(t_end - t_start))
        
        predict_boxes = predictions["boxes"].to("cpu").numpy()
        predict_keypoints = predictions["keypoints"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()
        
        recognize_keypoints = []
        for p_box, p_point, p_score in zip(predict_boxes, predict_keypoints, predict_scores):
            if p_score < 0.9 or len(p_point) == 0: continue
            print(f"p_box： {p_box.tolist()}， p_point: {p_point.tolist()}")
            start_points = p_point.tolist()[0]
            end_points = p_point.tolist()[1]
            if len(start_points)==0 or len(end_points)==0: continue
            angle = angle_beween_points(start_points, end_points)
            if angle>-10 and angle<10: 
                arrow_type = "arrow_line_right"
            elif angle>80 and angle<100: 
                arrow_type = "arrow_line_down"
            elif angle>170 and angle<-170: 
                arrow_type = "arrow_line_left"
            else:
                arrow_type = "arrow_line_up"
                
            recognize_keypoints.append({
                "boxes": p_box.tolist(),
                "start": start_points,
                "end": end_points,
                "class": arrow_type,
                "angle": angle
            })
            print("----------------------------------------------------")
        # print(f"recognize_keypoints: {recognize_keypoints}")
        draw_res(img_path=image_path, res_shapes=recognize_keypoints, save_path=save_path)
        
        return recognize_keypoints
    
if __name__ == '__main__':
    recognize()
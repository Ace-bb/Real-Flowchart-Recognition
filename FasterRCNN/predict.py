import os
# os.chdir("/home/zzq")
# print (os.getcwd())#获得当前目录
# print (os.path.abspath('.'))#获得当前工作目录
# print (os.path.abspath('..'))#获得当前工作目录的父目录
# print (os.path.abspath(os.curdir))#获得当前工作目录
import time
import json
import cv2

import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
if os.getcwd().split("/")[-1]=="FasterRCNN":
    from network_files import FasterRCNN, FastRCNNPredictor, AnchorsGenerator
    from backbone import resnet50_fpn_backbone, MobileNetV2
    from draw_box_utils import draw_objs, draw_arrow_start_end_node
    from arrow_recognize.recognize_arrow import recognize_arrow_tips
else:
    from FasterRCNN.network_files import FasterRCNN, FastRCNNPredictor, AnchorsGenerator
    from FasterRCNN.backbone import resnet50_fpn_backbone, MobileNetV2
    from FasterRCNN.draw_box_utils import draw_objs, draw_arrow_start_end_node
    from FasterRCNN.arrow_recognize.recognize_arrow import recognize_arrow_tips
    
def create_model(num_classes):
    # mobileNetv2+faster_RCNN
    # backbone = MobileNetV2().features
    # backbone.out_channels = 1280
    #
    # anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
    #                                     aspect_ratios=((0.5, 1.0, 2.0),))
    #
    # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
    #                                                 output_size=[7, 7],
    #                                                 sampling_ratio=2)
    #
    # model = FasterRCNN(backbone=backbone,
    #                    num_classes=num_classes,
    #                    rpn_anchor_generator=anchor_generator,
    #                    box_roi_pool=roi_pooler)

    # resNet50+fpn+faster_RCNN
    # 注意，这里的norm_layer要和训练脚本中保持一致
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    # get devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_model(num_classes=5)

    # load train weights
    weights_path = "./save_weights/2023-09-03-06-42-34/resNetFpn-model-20.pth"
    assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)

    # read class_indict
    label_json_path = './Datasets/v5/VOCdevkit/VOC2012/ImageSets/Main/classes.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as f:
        class_dict = json.load(f)

    category_index = {str(v): str(k) for k, v in class_dict.items()}

    # load image
    original_img = Image.open("./test/11.jpg").convert("RGB")

    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
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
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()

        if len(predict_boxes) == 0:
            print("没有检测到任何目标!")

        plot_img = draw_objs(original_img,
                            predict_boxes,
                            predict_classes,
                            predict_scores,
                            category_index=category_index,
                            box_thresh=0.5,
                            line_thickness=3,
                            font='arial.ttf',
                            font_size=20)
        plt.imshow(plot_img)
        plt.show()
        # 保存预测的图片结果
        plot_img.save("./test/test_result.jpg")

def flowchat_recognize(img_path, save_path,
                    model_path="./save_weights/v3OnlyLine2/2024-01-05-05-21-18/resNetFpn-model-19.pth",
                    classed_file = "./Datasets/v3OnlyLine/VOCdevkit/VOC2012/ImageSets/Main/classes.json"
                    ):
    # get devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    
    # read class_indict
    label_json_path = classed_file
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as f:
        class_dict = json.load(f)

    category_index = {str(v): str(k) for k, v in class_dict.items()}
    print(f"category_index: {category_index}")
    # create model
    model = create_model(num_classes=len(class_dict)+1)

    # load train weights
    weights_path = model_path
    assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)

    # load image
    original_img = Image.open(img_path).convert("RGB")

    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    res_shapes = list()
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
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()
        print(predict_boxes)
        print(f"predict_classes: {predict_classes}")
        if len(predict_boxes) == 0:
            print("没有检测到任何目标!")

        for p_box, p_class, p_score in zip(predict_boxes, predict_classes, predict_scores):
            print(f"p_class: {p_class}, p_score: {p_score}, p_box: {p_box}")
            # print(type(p_class))
            if p_class == 5 and p_score>0.2: 
                res_shapes.append({
                    "box": p_box.tolist(),
                    "class": category_index[str(p_class)]
                })
            if p_score < 0.90: continue
            res_shapes.append({
                "box": p_box.tolist(),
                "class": category_index[str(p_class)]
            })
        print(res_shapes)
        # 识别箭头，每个箭头有start-end表示
        # arrow_tips = recognize_arrow_tips(img_path, predict_boxes)
        
        
        plot_img = draw_objs(original_img,
                            predict_boxes,
                            predict_classes,
                            predict_scores,
                            category_index=category_index,
                            box_thresh=0.5,
                            line_thickness=3,
                            font='arial.ttf',
                            font_size=20)
        # plot_img = draw_arrow_start_end_node(plot_img, arrow_tips)
        # plt.imshow(plot_img)
        # plt.show()
        # # 保存预测的图片结果
        img_name = img_path.split("/")[-1].split(".")[0]
        plot_img.save(f"{save_path}/res_{img_name}.jpg")
    return res_shapes

if __name__ == '__main__':
    # main()
    flowchat_recognize(img_path="../images/cover_image_9.png", save_path="../results/test")

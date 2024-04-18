from conf.setting import *

import json
import os
import easyocr
import torch
from torchvision import transforms
from PIL import Image
import threading

from cnocr import CnOcr
import cv2
from graph import Graph
# from model.shape_classifier import ShapeClassifier
from node import Node, Arrow
from FasterRCNN.predict import flowchat_recognize, create_model
from FasterRCNN.arrow_recognize.recognize_arrow import recognize_arrow_tips
from FasterRCNN.arrow_recognize.arrow import recognize_arrow, recognize_arrow_v2
# from FasterRCNN.arrow_recognize.arrow_keypoint_predict import recognize
# from conf.flowchartTreeNode import FlowChartTreeNode as TreeNode

class FlowchartRecognition(object):
    def __init__(self, recognize_model, category_index, ocr_model, recognize_lock, cnocr_lock, file_create_lock, env_name="flowchart") -> None:
        self.RESULTS_PATH = "./results"
        self.selected_image = ""
        self.models_path = "./FasterRCNN/save_weights/V15ArrowMix/resNetFpn-model-15.pth"
        self.classed_file = "FasterRCNN/save_weights/V15ArrowMix/classes.json"
        self.arrow_model_path = "./FasterRCNN/save_weights/arrow_keypoint/keypointsrcnn_weights_8.pth"
        self.env_name = env_name
        self.res_save_path = "./results"
        self.recognize_model = recognize_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.category_index = category_index
        self.data_transform = transforms.Compose([transforms.ToTensor()])
        # self.ocr = CnOcr(rec_model_name='densenet_lite_136-gru', rec_model_backend="pytorch", det_model_name="db_resnet34", det_model_backend="pytorch")
        self.ocr = ocr_model
        
        self.recognize_lock = recognize_lock
        self.cnocr_lock = cnocr_lock
        self.file_create_lock = file_create_lock
        
        # self.init_flowchart_recognize_model()
    
    def init_flowchart_recognize_model(self):
        # get devices
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # read class_indict
        assert os.path.exists(self.classed_file), "json file {} dose not exist.".format(self.classed_file)
        with open(self.classed_file, 'r') as f:
            class_dict = json.load(f)
        self.category_index = {str(v): str(k) for k, v in class_dict.items()}
        
        # create model
        self.recognize_model = create_model(num_classes=len(class_dict)+1)
        # load train weights
        weights_path = self.models_path
        assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
        weights_dict = torch.load(weights_path, map_location='cpu')
        weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
        self.recognize_model.load_state_dict(weights_dict)
        self.recognize_model.to(self.device)
        self.recognize_model.eval()  # 进入验证模式
        
        
    def __get_results_path(self):
        results_dir = os.listdir(self.RESULTS_PATH)
        n = len(results_dir) + 1

        while(True):
            new_dir = "results_"+str(n)
            if(os.path.isdir(self.RESULTS_PATH + new_dir)):
                n += 1
            else:
                break
        return self.RESULTS_PATH + new_dir
    
    def genererate_save_path(self, file_name:str):
        folder_names = file_name.replace(".png","").replace(".jpg","").split("/")
        save_path = self.RESULTS_PATH
        for folder in folder_names:
            save_path = f"{save_path}/{folder}"
        self.file_create_lock.acquire()
        if not os.path.exists(save_path): os.makedirs(save_path)
        self.file_create_lock.release()
        return save_path
    
    def time_synchronized(self):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        return time.time()
    
    def train_shape_model(self,):
        ...

    def start_recognize_flowchart(self, img_path):
        # load image
        original_img = Image.open(img_path).convert("RGB")

        # from pil image to tensor, do not normalize image
        # data_transform = transforms.Compose([transforms.ToTensor()])
        img = self.data_transform(original_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        res_shapes = list()
        try:
            self.recognize_lock.acquire()
            with torch.no_grad():
                # init
                img_height, img_width = img.shape[-2:]
                init_img = torch.zeros((1, 3, img_height, img_width), device=self.device)
                self.recognize_model(init_img)
                predictions = self.recognize_model(img.to(self.device))[0]
            self.recognize_lock.release()
        except Exception as e:
            console.log(f"[bold red]recognize_model:{e}")
            self.recognize_lock.release()
            return []

        predict_boxes = predictions["boxes"].to("cpu").numpy()
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()
        if len(predict_boxes) == 0:
            return []

        for p_box, p_class, p_score in zip(predict_boxes, predict_classes, predict_scores):
            # if p_class==5 and p_score>0.2:
            #     res_shapes.append({
            #         "box": p_box.tolist(),
            #         "class": self.category_index[str(p_class)]
            #     })
            if p_score < 0.60: continue
            res_shapes.append({
                "box": p_box.tolist(),
                "class": self.category_index[str(p_class)]
            })
        return res_shapes
            
    
    def recognize_shape_arrow(self, img_path, save_path, text_shapes):
        """识别图片中的图形和箭头。识别图形的模型为FasterRCNN使用4种基本类别进行训练，
        只识别基础的process、decision、start_end、scan4种类型
        箭头的识别使用cv2的轮廓识别，无法识别一到多、多到多的关系
        问题在于无法区分轮廓点中哪些是起点和终点    

        Args:
            img_path (str): 图片的路径
            save_path (str): 识别信息保存路径
            text_shapes (list): 图片中已经识别的文字的信息，包括bbox和文字内容

        Returns:
            _type_: _description_
        """        
        flowchart_shapes = flowchat_recognize(img_path, save_path, 
                                            model_path = self.models_path,
                                            classed_file = self.classed_file)
        cover_shapes = list()
        for shape in flowchart_shapes: cover_shapes.append(shape['box'])
        for text in text_shapes: cover_shapes.append([text.coordinate[0],text.coordinate[2],text.coordinate[1],text.coordinate[3]])
        cover_image = cv2.imread(img_path)
        for cover in cover_shapes:
            cv2.rectangle(cover_image, (int(cover[0]), int(cover[1])), (int(cover[2]), int(cover[3])), (255,255,255), -1)
        cv2.imwrite(f"{save_path}/cover_image_{img_path.split('/')[-1].split('.')[0]}.png", cover_image)
        arrow_tips = recognize_arrow(img_path, cover_shapes, save_path=save_path, img_name=img_path.split('/')[-1].split('.')[0])
        # arrow_tips = recognize(arrow_img_path=f"{save_path}/cover_image_{img_path.split('/')[-1].split('.')[0]}.png", model_path=self.arrow_model_path, save_path=self.save_result_path)
        # for arrow in arrow_tips: flowchart_shapes.append(arrow)
        flowchart_shape_nodes = list()
        for i in range(len(flowchart_shapes)):
            shape = flowchart_shapes[i]
            flowchart_shape_nodes.append(Node(i, coordinate=[int(shape['box'][0]),int(shape['box'][2]),int(shape['box'][1]),int(shape['box'][3])], 
                                            text="", class_shape=shape['class']))
        flowchart_arrow_nodes = list()
        for arrow in arrow_tips:
            flowchart_arrow_nodes.append(Arrow(start_point=[int(arrow['start'][0]),int(arrow['start'][1])], 
                                            end_point=[int(arrow['end'][0]),int(arrow['end'][1])], 
                                            edge_class=arrow['class']))
        return flowchart_shape_nodes, flowchart_arrow_nodes

    def recognize_shape_arrow_v2(self, img_path, save_path, text_shapes):
        """识别图片中的图形和箭头。识别图形的模型为FasterRCNN使用5种类别进行训练，
        只识别基础的process、decision、start_end、scan、arrow5中类型
        箭头只识别箭头的顶部，不识别其中的线。如此相当于知道了箭头的终点，线头轮廓中除去终点附近的点就都是起点
        
        Args:
            img_path (str): 图片的路径
            save_path (str): 识别信息保存路径
            text_shapes (list): 图片中已经识别的文字的信息，包括bbox和文字内容

        Returns:
            _type_: _description_
        """ 
        # print(f"img_path: {img_path}")
        flowchart_shapes = self.start_recognize_flowchart(img_path)
        # print(f"flowchart_shapes: {flowchart_shapes}")
        cover_shapes = [shape['box'] for shape in flowchart_shapes if shape['class']!="arrow"]
        if len(cover_shapes)==0: 
            with open("results/error/error.txt", 'a', encoding="utf-8") as f:
                f.write(img_path+'\n')
            return [],[]
        all_x, all_y = list(), list()
        for [x1,y1,x2,y2] in cover_shapes: all_x.extend([int(x1),int(x2)])
        for [x1,y1,x2,y2] in cover_shapes: all_y.extend([int(y1),int(y2)])
        min_x, min_y, max_x, max_y = min(all_x), min(all_y), max(all_x), max(all_y)
        
        for text in text_shapes: cover_shapes.append([text.coordinate[0],text.coordinate[2],text.coordinate[1],text.coordinate[3]])
        arrow_tip_shapes =  [item['box'] for item in flowchart_shapes if item['class']=="arrow"]
        try:
            arrow_tips = recognize_arrow_v2(img_path, [min_x,max_x,min_y,max_y], cover_shapes, save_path=save_path, img_name=img_path.split('/')[-1].split('.')[0], arrow_tip_shapes=arrow_tip_shapes)
        except:
            arrow_tips  = []
        flowchart_shape_nodes = list()
        for i in range(len(flowchart_shapes)):
            shape = flowchart_shapes[i]
            if shape['class'] == "arrow": continue
            flowchart_shape_nodes.append(Node(i, coordinate=[int(shape['box'][0]),int(shape['box'][2]),int(shape['box'][1]),int(shape['box'][3])], 
                                            text="", class_shape=shape['class']))
        flowchart_arrow_nodes = list()
        for arrow in arrow_tips:
            flowchart_arrow_nodes.append(Arrow(start_point=[int(arrow['start'][0]),int(arrow['start'][1])], 
                                            end_point=[int(arrow['end'][0]),int(arrow['end'][1])], 
                                            edge_class=arrow['class']))
        
        return flowchart_shape_nodes, flowchart_arrow_nodes
        
    def recognize_flowchart(self, image_path, img_name, score_threshold=0.6, bbox_size_threshold=10):
        """开始识别流程图

        Args:
            image_path (str): 图片路径
            img_name (str): 图片名
            score_threshold (float, optional): 最低分数. Defaults to 0.6.
            bbox_size_threshold (int, optional): 边框大小阈值. Defaults to 10.

        Returns:
            result_nodes: 节点数组
            {
                "id": node.id,
                "Name": node.get_text(),
                "coordinate": node.get_coordinate(),
                "top": node.get_center_point_top(),
                "left": node.get_center_point_left(),
                "size": node.get_node_size()
            }
            {
                "id": f"{arrow.start_node}--{arrow.end_node}",
                "sourceNode": str(arrow.start_node) + '',
                "targetNode": str(arrow.end_node) + '',
                "source": arrow.source,
                "target": arrow.target,
            }
            result_arrows: 箭头数组
        """
        self.res_save_path = self.genererate_save_path(img_name)
        #Get the image
        # print(f"Detecting text: {image_path}")
        text_nodes = self.recognize_text_v2(image_path, score_threshold, bbox_size_threshold)
        # print(f"\nchinese ocr texts :{text_nodes}\n")
        shape_nodes, arrow_nodes = self.recognize_shape_arrow_v2(image_path, self.res_save_path, text_nodes)
        
        # print(f"shape nodes is:{shape_nodes}\n")
        # print(f"arrow nodes is:{arrow_nodes}\n")
        self.save_recognised_imgs(img_path=image_path, annotations=shape_nodes, textAnnots=text_nodes, img_name=image_path.split('/')[-1])
        # Generate flowchart graph
        graph = Graph(text_nodes, shape_nodes, arrow_nodes)
        # flow = graph.generate_graph()
        graph.collapse_nodes_arrow()
        # print(f"flow: {flow}")
        # return graph.nodes, graph.arrow_nodes
        return self.generate_shapes_arrow_result(graph)
    
    def merge_lines(self, lines:list()):
        merged_lines = list()
        merged_ids = set()
        for i, l1 in enumerate(lines):
            for j, l2 in enumerate(lines):
                if j<=i: continue
                if tools.calculate_distance(l1['p1'], l2['p1']) < 10:
                    merged_lines.append({"p1":l1['p2'], "p2": l2['p2']})
                    merged_ids.update([i,j])
                elif tools.calculate_distance(l1['p1'], l2['p2']) < 10:
                    merged_lines.append({"p1":l1['p2'], "p2": l2['p1']})
                    merged_ids.update([i,j])
                elif tools.calculate_distance(l1['p2'], l2['p1']) < 10:
                    merged_lines.append({"p1":l1['p1'], "p2": l2['p2']})
                    merged_ids.update([i,j])
                elif tools.calculate_distance(l1['p2'], l2['p2']) < 10:
                    merged_lines.append({"p1":l1['p1'], "p2": l2['p1']})
                    merged_ids.update([i,j])
        left_lines = list()
        for k, l1 in enumerate(lines):
            if k in merged_ids: continue
            left_lines.append(l1)
        if len(merged_lines)==0: return lines
        elif len(left_lines)==0: return merged_lines
        else:
            merged_lines.extend(left_lines)
            return self.merge_lines(merged_lines)
    
    def calculate_direction(self, s_p, e_p):
        if abs(s_p[0]-e_p[0]) > abs(s_p[1]-e_p[1]): # 横向
            if e_p[0] >= s_p[0]: # 向右
                return "arrow_line_right"
            else: return "arrow_line_left"
        else: # 纵向
            if e_p[1]>=s_p[1]: return "arrow_line_down"
            else: return "arrow_line_up"
            
    def merge_arrow_lines(self, arrows, lines):
        merged_lines = list()
        merged_line_ids = set()
        main_dir = {
            "arrow_line_down": 0,
            "arrow_line_right": 0,
            "arrow_line_left": 0,
            "arrow_line_up": 0
        }
        for ar in arrows:
            for i, line in enumerate(lines):
                if i in merged_line_ids: continue
                if tools.calculate_distance(ar, line['p1'])<10:
                    ppd =  self.calculate_direction(line['p2'], line['p1'])
                    main_dir[ppd]+=1
                    merged_lines.append({ "start": line['p2'], "end": line['p1'], "class": ppd })
                    merged_line_ids.add(i)
                    
                elif tools.calculate_distance(ar, line['p2'])<10:
                    ppd = self.calculate_direction(line['p1'], line['p2'])
                    main_dir[ppd]+=1
                    merged_lines.append({ "start": line['p1'], "end": line['p2'], "class": ppd })
                    merged_line_ids.add(i)
                    
        left_lines =  list()
        direction = max(main_dir.keys(), key=lambda x: main_dir[x])
        for j, line in lines:
            if j in merged_line_ids: continue
            if direction=="arrow_line_down":
                merged_lines.append({ "start": line['p1'] if line['p2'][1]>=line['p1'][1] else line['p2'], 
                                    "end": line['p2'] if line['p2'][1]>=line['p1'][1] else line['p1'],
                                    "class": "arrow_line_down"})
            else:
                merged_lines.append({ "start": line['p1'] if line['p2'][0]>=line['p1'][0] else line['p2'], 
                                    "end": line['p2'] if line['p2'][0]>=line['p1'][0] else line['p1'],
                                    "class": "arrow_line_down"})
        return merged_lines
    
    def recognize_flowchart_lines(self, img_path, img_shape, bonding_boxes, save_path="", img_name="img_name", arrow_tip_shapes=[]):
        image = cv2.imread(img_path)
        img_h, img_w = image.shape[:2]
        cv2.rectangle(image, (0, 0), (img_w, img_shape[2]), (255, 255, 255), -1)
        cv2.rectangle(image, (0, 0), (img_shape[0], img_h), (255, 255, 255), -1)
        cv2.rectangle(image, (0, img_shape[3]), (img_w, img_h), (255, 255, 255), -1)
        cv2.rectangle(image, (img_shape[1], 0), (img_w, img_h), (255, 255, 255), -1)
        for box in bonding_boxes:
            cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 255), -1)
        cv2.imwrite(f"{save_path}/only_line{img_name}.png", image)
        
        res_shapes = self.start_recognize_flowchart(img_path=f"{save_path}/only_line{img_name}.png")
        recognized_shapes = list()
        recognized_arrows = list()
        for shape in res_shapes:
            xmin, ymin, xmax,ymax = shape['box']
            if shape['class'] == "line":
                
                if (xmax-xmin)>=(ymax-ymin):
                    recognized_shapes.append({
                        "p1": [xmin, (ymin+ymax)//2],
                        "p2": [xmax, (ymin+ymax)//2]
                    })
                else:
                    recognized_shapes.append({
                        "p1": [(xmin+xmax)//2, ymin],
                        "p2": [(xmin+xmax)//2, ymax]
                    })
            elif shape['class']=='arrow':
                recognized_arrows.append([(xmin+xmax)//2, (ymin+ymax)//2])
                
        merged_lines = self.merge_lines(recognized_shapes)
        merged_arrow_lines = self.merge_arrow_lines(recognized_arrows, merged_lines)
        
        return merged_arrow_lines
    
    def construct_text_nodes(self, ocr_texts):
        text_nodes = list()
        for item in ocr_texts:
            text_nodes.append(Node(coordinate=[int(item['box'][0]), int(item['box'][2]), int(item['box'][1]), int(item['box'][5])], text=item['text']))
        return text_nodes
        
    def recognize_text(self, image_path):
        """"""
        reader = easyocr.Reader(['ch_sim','en']) # this needs to run only once to load the model into memory
        result = reader.readtext(image_path, paragraph=False, text_threshold=0.7, low_text=0.4)
        text_nodes = list()
        for i in range(len(result)):
            item = result[i]
            text_nodes.append(Node(idx=i, coordinate=[int(item[0][0][0]), int(item[0][1][0]), int(item[0][0][1]), int(item[0][2][1])], text=item[1]))
        return text_nodes
    
    def recognize_text_v2(self, image_path, score_threshold=0.6, bbox_size_threshold=10):
        """"""
        self.cnocr_lock.acquire()
        try:
            result = self.ocr.ocr(image_path, cls=True)
        except Exception as e:
            console.log(f"[bold red]{e}")
            result=[]
        self.cnocr_lock.release()
        
        text_nodes = list()
        # console.log(len(result))
        for i in range(len(result[0])):
            item = result[0][i]
            # text_nodes.append(Node(idx=i, 
            #                     coordinate=[
            #                         int(item['position'][0][0]), 
            #                         int(item['position'][2][0]), 
            #                         int(item['position'][0][1]), 
            #                         int(item['position'][2][1]),], 
            #                     text=item['text']))
            text_nodes.append(Node(idx=i, 
                                coordinate=[
                                    int(item[0][0][0]), 
                                    int(item[0][2][0]), 
                                    int(item[0][0][1]), 
                                    int(item[0][2][1]),], 
                                text=item[1][0]))
        return text_nodes
    
    def generate_shapes_arrow_result(self, graph:Graph):
        result_nodes = list()
        result_arrows = list()
        graph_nodes = graph.get_nodes()
        graph_edges = graph.arrow_nodes
        for node in graph_nodes:
            result_nodes.append({
                "id": node.id,
                "Name": node.get_text(),
                "coordinate": node.get_coordinate(),
                "top": node.get_center_point_top(),
                "left": node.get_center_point_left(),
                "size": node.get_node_size()
            })
        for arrow in graph_edges:
            if arrow.start_node == "" or arrow.end_node == "": continue
            result_arrows.append({
                "id": f"{arrow.start_node}--{arrow.end_node}",
                "sourceNode": str(arrow.start_node) + '',
                "targetNode": str(arrow.end_node) + '',
                "source": arrow.source,
                "target": arrow.target,
                "label": arrow.edge_label
            })
        # print(f"result_nodes: {result_nodes}")
        # print(f"result_arrows: {result_arrows}")
        return result_nodes, result_arrows
    
    def generate_result(self, graph:Graph, img_name):
        start_node_id = graph.first_state
        graph_node = graph.get_nodes()
        adj_list = graph.adj_list
        results_nodes_json = list()
        # start_node = TreeNode(nid=start_node_id, pid=-1, Name=graph_node[start_node_id].get_text())
        added_ids = []
        results_nodes_json.append({
                                    "id": start_node_id,
                                    "parent": -1,
                                    "Name": graph_node[start_node_id].get_text(),
                                    "coordinate": graph_node[start_node_id].get_coordinate(),
                                    "top": graph_node[start_node_id].get_center_point_top(),
                                    "left": graph_node[start_node_id].get_center_point_left()
                                })
        added_ids.append(start_node_id)
        for key in adj_list.keys():
            if "arrow" in graph_node[key].get_class() or len(adj_list[key]) == 0 or graph_node[key].get_text() == "": continue
            parent_node = graph_node[key]
            node_childs_edges = adj_list[key]
            for edge_id in node_childs_edges:
                sub_node = graph_node[adj_list[edge_id][0]]
                results_nodes_json.append({
                    "id": adj_list[edge_id][0],
                    "parent": key,
                    "Name": sub_node.get_text(),
                    "coordinate": sub_node.get_coordinate(),
                    "top": sub_node.get_center_point_top(),
                    "left": sub_node.get_center_point_left()
                })
                added_ids.append(adj_list[edge_id][0]) 
        
        for i in range(len(graph_node)):
            if i in added_ids or graph_node[i].get_text()=="": continue
            # print(f"graph node: {graph_node[i].get_text()}")
            results_nodes_json.append({
                "id": i,
                "parent": None,
                "Name": graph_node[i].get_text(),
                "coordinate": graph_node[i].get_coordinate(),
                "top": graph_node[i].get_center_point_top(),
                "left": graph_node[i].get_center_point_left()
            })
        
        #     for sub_node_id in flow[key]:
        #         temp_sub_node = TreeNode(nid=sub_node_id, pid=key, Name=graph_node[sub_node_id].get_text())
        with open(f"{self.res_save_path}/recognition_res_{img_name}.json", 'w', encoding='utf-8') as f:
            json.dump(results_nodes_json, f, ensure_ascii=False)
        
        return results_nodes_json

    def save_recognised_imgs(self, img_path, annotations, textAnnots=None, img_name = "save_recognised_imgs"):
        img = cv2.imread(img_path)
        # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
        # cv2.imshow('src',img)
        for annot in annotations:
            # if annot.class_shape == 'process':
            cv2.rectangle(img, 
                            (annot.coordinate[0], annot.coordinate[2]), 
                            (annot.coordinate[1], annot.coordinate[3]), (0,255,0), 2)
            # elif annot.class_shape == 'process':
                # ...
        if textAnnots != None:
            for annot in textAnnots:
                cv2.rectangle(img, 
                            (annot.coordinate[0], annot.coordinate[2]), 
                            (annot.coordinate[1], annot.coordinate[3]), (0,0,255), 2)
                
        # img_name = img_path.split('/')[-1]
        cv2.imwrite(f"{self.res_save_path}/{img_name}", img)
    
    def draw_recognized_node_edges(self, nodes, edges, img_path):
        img = cv2.imread(img_path)
        id2coordinate = {}
        dp = {
            "left": lambda p: (p[0], (p[2] + p[3])//2),
            "right": lambda p: (p[1], (p[2] + p[3])//2),
            "top": lambda p: ((p[0]+p[1])//2, p[2]),
            "bottom": lambda p: ((p[0]+p[1])//2, p[3])
        }
        for node in nodes:
            cv2.rectangle(img, 
                            (node['coordinate'][0], node['coordinate'][2]), 
                            (node['coordinate'][1], node['coordinate'][3]), (0,0,255), 2)
            
            cv2.putText(img, str(node['id']), ((node['coordinate'][0]+node['coordinate'][1])//2,( node['coordinate'][2]+node['coordinate'][3])//2), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 0, 255), 4)
            id2coordinate[str(node['id'])] = node['coordinate']
        
        for edge in edges:
            cv2.line(img, dp[edge["source"]](id2coordinate[edge["sourceNode"]]), dp[edge["target"]](id2coordinate[edge["targetNode"]]), (0,0,255), 3)
        cv2.imwrite(f"{self.res_save_path}/draw-node-edge.png", img)
        
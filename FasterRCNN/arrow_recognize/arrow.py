import math
import cv2
import numpy as np


def get_filter_arrow_image(threslold_image):
    blank_image = np.zeros_like(threslold_image)

    # dilate image to remove self-intersections error
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    threslold_image = cv2.dilate(threslold_image, kernel_dilate, iterations=1)
    cv2.imwrite("thresh_image——dilate.png", thresh_image)

    contours, hierarchy = cv2.findContours(threslold_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is not None:

        threshold_distnace = 1000

        for cnt in contours:
            hull = cv2.convexHull(cnt, returnPoints=False)
            defects = cv2.convexityDefects(cnt, hull)

            if defects is not None:
                for i in range(defects.shape[0]):
                    start_index, end_index, farthest_index, distance = defects[i, 0]

                    # you can add more filteration based on this start, end and far point
                    # start = tuple(cnt[start_index][0])
                    # end = tuple(cnt[end_index][0])
                    # far = tuple(cnt[farthest_index][0])

                    if distance > threshold_distnace:
                        cv2.drawContours(blank_image, [cnt], -1, 255, -1)

        return blank_image
    else:
        return None


def get_length(p1, p2):
    line_length = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
    return line_length


def get_max_distace_point(cnt):
    max_distance = 0
    max_points = None
    for [[x1, y1]] in cnt:
        for [[x2, y2]] in cnt:
            distance = get_length((x1, y1), (x2, y2))
            # # print(distance)
            if distance > max_distance:
                max_distance = distance
                max_points = [(x1, y1), (x2, y2)]

    return max_points


def angle_beween_points(a, b):
    if b[0] - a[0] == 0:
        if b[1] - a[1] > 0:
            arrow_slope = float("inf")
        else:
            arrow_slope = -float("inf")
    else:
        arrow_slope =  (b[1] - a[1]) / (b[0] - a[0])
    arrow_angle = math.degrees(math.atan(arrow_slope))
    return arrow_angle

def preprocess(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
    img_canny = cv2.Canny(img_blur, 50, 50)
    kernel = np.ones((3, 3))
    img_dilate = cv2.dilate(img_canny, kernel, iterations=2)
    img_erode = cv2.erode(img_dilate, kernel, iterations=1)
    return img_erode

def get_arrow_info(arrow_image):
    arrow_info_image = cv2.cvtColor(arrow_image.copy(), cv2.COLOR_GRAY2BGR)
    contours, hierarchy = cv2.findContours(arrow_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    res_arrow_info = []
    if hierarchy is not None: 

        for cnt in contours:
            if len (cnt) <=1: continue
            # draw single arrow on blank image
            blank_image = np.zeros_like(arrow_image)
            cv2.drawContours(blank_image, [cnt], -1, 255, -1)

            s_p, e_p = get_max_distace_point(cnt)

            angle = angle_beween_points(s_p, e_p)
            lenght = get_length(s_p, e_p)

            cv2.line(arrow_info_image, s_p, e_p, (0, 255, 255), 1)

            cv2.circle(arrow_info_image, s_p, 2, (255, 0, 0), 3)
            cv2.circle(arrow_info_image, e_p, 2, (255, 0, 0), 3)

            cv2.putText(arrow_info_image, "start angle : {0:0.2f}".format(angle),
                        s_p, cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 255), 1)
            cv2.putText(arrow_info_image, "end lenght : {0:0.2f}".format(lenght),
                        (e_p[0], e_p[1] + 20), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 255), 1)

            if angle>-70 and angle<70: 
                res_arrow_info.append({"start": s_p, "end": e_p, "class": "arrow_line_right"})
            elif angle>20 and angle<160: 
                res_arrow_info.append({"start": s_p, "end": e_p, "class": "arrow_line_down"})
            elif angle>170 and angle<-170: 
                res_arrow_info.append({"start": s_p, "end": e_p, "class": "arrow_line_left"})
            else:
                res_arrow_info.append({"start": s_p, "end": e_p, "class": "arrow_line_up"})
                
        return arrow_info_image, res_arrow_info
    else:
        return None, None

from numpy import unique
from numpy import where
import numpy as np
from sklearn.datasets import make_classification
from sklearn.cluster import DBSCAN
import copy
def filter_arrow_cnt_nodes(contours):
    filtered_nodes = list()
    cnt_X = list()
    for [[x1, y1]] in contours:
        cnt_X.append([x1, y1])
    model = DBSCAN(eps=5, min_samples=2)
    yhat = model.fit_predict(cnt_X)
    # 检索唯一群集
    clusters = unique(yhat)
    for cluster in clusters:
    # 获取此群集的示例的行索引
        row_ix = where(yhat == cluster)
        # # print([cnt_X[ix] for ix in row_ix])
        c_X = np.mean([cnt_X[ix][0] for ix in row_ix[0]])
        c_Y = np.mean([cnt_X[ix][1] for ix in row_ix[0]])
        filtered_nodes.append([int(c_X), int(c_Y)])
    return filtered_nodes

def get_contain_arrow_tips(arrow_nodes, arrow_tips):
    """获取轮廓中包含的箭头

    Args:
        contour (_type_): _description_
        arrow_tips (_type_): _description_
    """  
    contain_tips = list()
    arrow_end_tips = list()  
    for point in arrow_nodes:
        for arrow in arrow_tips:
            if arrow in contain_tips: continue
            if (point[0] >= arrow[0] and point[1]>=arrow[1]) and (point[0] <= arrow[2] and point[1]<=arrow[3]):
                contain_tips.append(arrow)
                arrow_end_tips.append(point)
                break
    
    return arrow_end_tips

def get_farthest_point(p, points):
    max_dis = 0
    farthest_point = p
    for point in points:
        distance = get_length((p[0], p[1]), (point[0], point[1]))
        if distance> max_dis:
            max_dis = distance
            farthest_point = point
    return farthest_point

def arrow_direction(s_point, e_point): #计算箭头的主方向
    if s_point==None or e_point==None: return None
    if (e_point[0]-s_point[0])>=0 and  (e_point[1]-s_point[1])>=0:
        if abs(e_point[0]-s_point[0]) - abs(e_point[1]-s_point[1]) >0:
            return "arrow_line_right"
        else:
            return "arrow_line_down"
    elif (e_point[0]-s_point[0])>=0 and  (e_point[1]-s_point[1])<0:
        if abs(e_point[0]-s_point[0]) - abs(e_point[1]-s_point[1]) >0:
            return "arrow_line_right"
        else:
            return "arrow_line_up"
    elif (e_point[0]-s_point[0])<0 and  (e_point[1]-s_point[1])>=0:
        if abs(e_point[0]-s_point[0]) > abs(e_point[1]-s_point[1]):
            return "arrow_line_left"
        else:
            return "arrow_line_down"
    else:
        if abs(e_point[0]-s_point[0]) > abs(e_point[1]-s_point[1]):
            return "arrow_line_left"
        else:
            return "arrow_line_up"
    
def get_main_direction_farthest_point(end_points, all_points, direction):
    if direction not in ["arrow_line_down", "arrow_line_up", "arrow_line_right", "arrow_line_left"]:
        raise
    max_dis = 0
    farthest_point = None
    for e_p in end_points:
        for s_p in all_points:
            if direction == "arrow_line_down" or direction=="arrow_line_up":
                distance = abs(s_p[1] - e_p[1])
            elif direction == "arrow_line_right" or direction=="arrow_line_left":
                distance = abs(s_p[0] - e_p[0])
            if distance>max_dis:
                max_dis = distance
                farthest_point = s_p
    
    return farthest_point


def get_main_direction(main_directions):
    tmp_main_direction = [{"direction": k, "num": main_directions[k]} for k  in main_directions.keys()]
    tmp_main_direction.sort(key=lambda x: x['num'], reverse=True)
    if tmp_main_direction[0]['num'] == 0:
        return 'arrow_line_down'
    else:
        return tmp_main_direction[0]["direction"]

# def get_main_direction_start_point(end_points, all_points, direction):
#     if direction not in ["arrow_line_down", "arrow_line_up", "arrow_line_right", "arrow_line_left"]:
#         raise
#     max_dis = 0
#     farthest_point = None
#     min_y = min([node[1] for node in all_points])
#     max_y = max([node[1] for node in all_points])
#     mid_y = (min_y+max_y)//2
#     tmp_start_nodes = 
#     return farthest_point

def find_start_end_no_tip(nodes):
    min_y = min([node[1] for node in nodes])
    max_y = max([node[1] for node in nodes])
    mid_y = (min_y+max_y)//2
    start_nodes = list(filter(lambda x: x[1]<mid_y and (mid_y-x[1])>(x[1]-min_y), nodes))
    end_nodes = list(filter(lambda x: x[1]>mid_y and (x[1]-mid_y)>(max_y-x[1]), nodes))
    # # print(f"find_start_end_no_tip---{len(start_nodes)}---start_nodes: {start_nodes}")
    # # print(f"find_start_end_no_tip---{len(end_nodes)}---start_nodes: {end_nodes}")
    sid, eid = [],[]
    for i,s_n in enumerate(start_nodes):
        for j, e_n in enumerate(start_nodes):
            if j<=i: continue
            if abs(s_n[0]-e_n[0])<12: sid.append(i)
    res_start_nodes = [node for i,node in enumerate(start_nodes) if i not in sid]
    for i,s_n in enumerate(end_nodes):
        for j, e_n in enumerate(end_nodes):
            if j<=i: continue
            if abs(s_n[0]-e_n[0])<12: eid.append(i)
    res_end_nodes = [node for i,node in enumerate(end_nodes) if i not in eid]
    
    return res_start_nodes, res_end_nodes

def get_arrow_info_v2(arrow_image, arrow_tips, img_path, save_path):
    """使用OpenCV的轮廓分析的方法，识别箭头
    已知图片中全部箭头的顶部，即每个箭头的终点
    只需要识别起点即可。

    Args:
        arrow_image (_type_): _description_
        arrow_tips (_type_): _description_

    Returns:
        _type_: _description_
    """    
    arrow_info_image = cv2.cvtColor(arrow_image.copy(), cv2.COLOR_GRAY2BGR)
    contours, hierarchy = cv2.findContours(arrow_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    res_arrow_info = []
    main_direction = {
        "arrow_line_down": 0,
        "arrow_line_right": 0,
        "arrow_line_left": 0,
        "arrow_line_up": 0
    }
    # breakpoint()
    if hierarchy is not None: 
        point_img = cv2.imread(img_path)
        contours_img = cv2.imread(img_path)
        multi_end_contours = list()
        for i, cnt in enumerate(contours):
            # print(f"contours：{contours}")
            for [p] in cnt:
                cv2.circle(contours_img, p, 3, (0, 0, 255), 3)
            if len (cnt) <=1: continue
            # # print(cnt)
            filtered_nodes = filter_arrow_cnt_nodes(cnt)
            if len(filtered_nodes)<=1: 
                # print("filtered_nodesfiltered_nodesfiltered_nodesfiltered_nodesfiltered_nodesfiltered_nodesfiltered_nodesfiltered_nodesfiltered_nodes")
                continue
            # for p in filtered_nodes:
            #     cv2.circle(point_img, p, 3, (0, 0, 255), 3)
            
            # draw single arrow on blank image
            # blank_image = np.zeros_like(arrow_image)
            # cv2.drawContours(blank_image, [cnt], -1, 255, -1)
            end_points = get_contain_arrow_tips(filtered_nodes, arrow_tips)
            if len(end_points) == 0: # 说明是不包箭头的边
                # # print("不包含箭头")
                # # print(f"{len(filtered_nodes)}---filtered_nodes: {filtered_nodes}")
                if len(filtered_nodes)==2 and (abs(filtered_nodes[0][0] - filtered_nodes[1][0]) > 12 or abs(filtered_nodes[0][1] - filtered_nodes[1][1]) > 12):# 说明是两点的边
                    if abs(filtered_nodes[0][0] - filtered_nodes[1][0]) > abs(filtered_nodes[0][1] - filtered_nodes[1][1]): # 说明是x轴横向的边
                        start_point = filtered_nodes[0] if filtered_nodes[0][0]<filtered_nodes[1][0] else filtered_nodes[1]
                        end_point = filtered_nodes[1] if filtered_nodes[0][0]<filtered_nodes[1][0] else filtered_nodes[0]
                        res_arrow_info.append({"start": start_point, "end": end_point, "class": 'arrow_line_right'})
                    # elif abs(filtered_nodes[0][0] - filtered_nodes[1][0]) < abs(filtered_nodes[0][1] - filtered_nodes[1][1]):
                    else:
                        start_point = filtered_nodes[0] if filtered_nodes[0][1]<filtered_nodes[1][1] else filtered_nodes[1]
                        end_point = filtered_nodes[1] if filtered_nodes[0][1]<filtered_nodes[1][1] else filtered_nodes[0]
                        res_arrow_info.append({"start": start_point, "end": end_point, "class": 'arrow_line_down'})
                    continue
                start_nodes, end_nodes = find_start_end_no_tip(filtered_nodes)
                # # print(f"不包含箭头: start_nodes:{start_nodes}")
                # # print(f"不包含箭头: end_nodes:{end_nodes}")
                for s_node in start_nodes:
                    cv2.circle(point_img, s_node, 3, (0, 0, 255), 3)
                    for e_node in end_nodes:
                        cv2.circle(point_img, e_node, 3, (0, 0, 255), 3)
                        res_arrow_info.append({"start": s_node, "end": e_node, "class": 'arrow_line_down'})
                
                continue
            # # print(f"end_points:{end_points}")
            if len(end_points)==1: #只有一个箭头，最远的点就是起点
                start_point = get_farthest_point(end_points[0], filtered_nodes)
                end_point = end_points[0]
                a_direction = arrow_direction(start_point, end_point)
                main_direction[a_direction] +=1
                res_arrow_info.append({"start": start_point, "end": end_point, "class": a_direction})
                
                # 记录该箭头的方向
            else: # 终止箭头大于等于两个，计算主方向最远的点，暂存，处理完全部一个终止箭头的再处理这部分
                multi_end_contours.append({
                    "filtered_nodes": filtered_nodes,
                    "end_points": end_points
                })
        cv2.imwrite(f"{save_path}/filtered_nodes_point.png", point_img)
        cv2.imwrite(f"{save_path}/all_contours.png", contours_img)
        direction = get_main_direction(main_direction)
        # # print(f"direction: {direction}")
        for cnt in multi_end_contours: # 有多个终止箭头的，可能是多对多的情况
            filtered_nodes = cnt['filtered_nodes']
            end_points = cnt['end_points']
            # start_point = get_main_direction_farthest_point(end_points,filtered_nodes, direction)
            start_nodes, _ = find_start_end_no_tip(filtered_nodes)
            # # print(f"filtered_nodes: {filtered_nodes}")
            # # print(f"end_points: {end_points}")
            for start_p in start_nodes:
                for end_p in end_points:
                    # # print(f"start_point: {start_point}--end_p: {end_p}")
                    a_direction = arrow_direction(start_point, end_p)
                    if a_direction!=None:
                        main_direction[a_direction] +=1
                        res_arrow_info.append({"start": start_p, "end": end_p, "class": a_direction})

        for arrow in res_arrow_info:
            s_p = arrow["start"]
            e_p = arrow["end"]

            cv2.line(arrow_info_image, s_p, e_p, (255, 255, 255), 1)
            cv2.circle(arrow_info_image, s_p, 4, (0, 0, 255), 2)
            cv2.circle(arrow_info_image, e_p, 4, (0, 0, 255), 2)
            cv2.putText(arrow_info_image, "S", s_p, cv2.FONT_HERSHEY_PLAIN, 1.8, (255, 255, 255), 2)
            cv2.putText(arrow_info_image, "E", (e_p[0], e_p[1] + 20), cv2.FONT_HERSHEY_PLAIN, 1.8, (255, 255, 255), 2)

        return arrow_info_image, res_arrow_info
    else:
        # print("-----------------------------None______________________")
        return None, None
    
def recognize_arrow(img_path, bonding_boxes, save_path="", img_name="img_name"):
    image = cv2.imread(img_path)
    for box in bonding_boxes:
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 255), -1)
    cv2.imwrite(f"{save_path}/only_line.png", image)
    
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # _, thresh_image = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY_INV)
    thresh_image = preprocess(img=image)
    # arrow_image = get_filter_arrow_image(thresh_image)
    arrow_image = thresh_image
    cv2.imwrite(f"{save_path}/black_back_bold_line.png", arrow_image)
    if arrow_image is not None:
        # cv2.imshow("arrow_image", arrow_image)
        arrow_info_image, arrow_info = get_arrow_info(arrow_image)
        # cv2.imshow("arrow_info_image", arrow_info_image)
        cv2.imwrite(f"{save_path}/arrow_label_image.png", arrow_info_image)
        
        return arrow_info
    return None

def recognize_arrow_v2(img_path, img_shape, bonding_boxes, save_path="", img_name="img_name", arrow_tip_shapes=[]):
    image = cv2.imread(img_path)
    img_h, img_w = image.shape[:2]
    cv2.rectangle(image, (0, 0), (img_w, img_shape[2]), (255, 255, 255), -1)
    cv2.rectangle(image, (0, 0), (img_shape[0], img_h), (255, 255, 255), -1)
    cv2.rectangle(image, (0, img_shape[3]), (img_w, img_h), (255, 255, 255), -1)
    cv2.rectangle(image, (img_shape[1], 0), (img_w, img_h), (255, 255, 255), -1)
    # image = image[img_shape[2]:img_shape[3], img_shape[0]:img_shape[1]]
    
    for box in bonding_boxes:
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 255), -1)
    cv2.imwrite(f"{save_path}/only_line{img_name}.png", image)
    arrow_image = preprocess(img=image)
    cv2.imwrite(f"{save_path}/black_back_bold_line.png", arrow_image)
    if arrow_image is not None:
        arrow_info_image, arrow_info = get_arrow_info_v2(arrow_image, arrow_tip_shapes, img_path, save_path)
        cv2.imwrite(f"{save_path}/arrow_label_image.png", arrow_info_image)
        return arrow_info
    return None

if __name__ == "__main__":
    img_path = "/home/libinbin/Server/Decision-Tree/flowchart_recognition/results/General_diagnosis/09体重增加/cover_gray_image_09体重增加.png"
    image = cv2.imread(img_path)

    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # _, thresh_image = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY_INV)
    thresh_image = preprocess(img=image)
    save_path = "./test_res"

    # arrow_image = get_filter_arrow_image(thresh_image)
    arrow_image = thresh_image
    cv2.imwrite(f"{save_path}/get_filter_arrow_image.png", arrow_image)
    if arrow_image is not None:
        # cv2.imshow("arrow_image", arrow_image)

        arrow_info_image, arrow_info = get_arrow_info_v2(arrow_image, [], img_path = img_path)
        # cv2.imshow("arrow_info_image", arrow_info_image)
        cv2.imwrite(f"{save_path}/arrow_info_image.png", arrow_info_image)
    
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
import math
import cv2
import numpy as np
from numpy import unique
from numpy import where
from sklearn.cluster import KMeans, DBSCAN

def DBSCAN_arrow(X):
    dbscan = DBSCAN(eps=2, min_samples=3) # 创建DBSCAN对象，设置半径和最小样本数
    labels = dbscan.fit_predict(X)
    for i in range(max(labels)+1):
        print(f"Cluster {i+1}: {list(X[labels==i])}")
    print(f"Noise: {list(X[labels==-1])}")

    
def KMeans_arrow(X):
    # print(f"KMeans_arrow: {X}")
    # 定义模型
    model = KMeans(n_clusters=2)
    # 模型拟合
    model.fit(X)
    # 为每个示例分配一个集群
    yhat = model.predict(X)
    # 检索唯一群集
    clusters = unique(yhat)
    # 为每个群集的样本创建散点图
    res = {}
    for cluster in clusters: res[f"{cluster}"] = X[where(yhat == cluster)].tolist()
    
    return res["0"] if len(res["0"]) >= len(res["1"]) else res["1"]

def get_filter_arrow_image(threslold_image):
    blank_image = np.zeros_like(threslold_image)

    # dilate image to remove self-intersections error
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    threslold_image = cv2.dilate(threslold_image, kernel_dilate, iterations=1)

    contours, hierarchy = cv2.findContours(threslold_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is not None:

        threshold_distnace = 100

        for cnt in contours:
            hull = cv2.convexHull(cnt, returnPoints=False)
            defects = cv2.convexityDefects(cnt, hull)

            if defects is not None:
                for i in range(defects.shape[0]):
                    start_index, end_index, farthest_index, distance = defects[i, 0]

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
            if distance > max_distance:
                max_distance = distance
                max_points = [(x1, y1), (x2, y2)]
    return max_points


def angle_beween_points(a, b):
    # arrow_slope = (a[0] - b[0]) / (a[1] - b[1])
    arrow_slope =  (b[1] - a[1]) / (b[0] - a[0])
    arrow_angle = math.degrees(math.atan(arrow_slope))
    return arrow_angle

def find_tip(points, convex_hull):
    length = len(points)
    indices = np.setdiff1d(range(length), convex_hull)
    for i in range(2):
        j = indices[i] + 2
        if j > length - 1:
            j = length - j
        if np.all(points[j] == points[indices[i - 1] - 2]):
            return tuple(points[j])
        
def get_arrow_info(arrow_image, save_path, img_name):
    # blank_image = np.zeros_like(arrow_image)
    # cv2.imwrite(f"{save_path}/get_arrow_info_blank_image_{img_name}.png", blank_image)

    # dilate image to remove self-intersections error
    # kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # arrow_image = cv2.dilate(arrow_image, kernel_dilate, iterations=1) # 对图片进行膨胀处处理
    # cv2.imwrite(f"{save_path}/get_arrow_info_dilate_{img_name}.png", arrow_image)
    # 轮廓检测
    contours, hierarchy = cv2.findContours(arrow_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"get_arrow_info:{len(contours)}")
    arrow_info = []
    
    if hierarchy is not None:
        for cnt in contours:
            # draw single arrow on blank image
            blank_image = np.zeros_like(arrow_image)
            cv2.drawContours(blank_image, [cnt], -1, 255, -1)

            point1, point2 = get_max_distace_point(cnt)
            # print(f"len: {len(cnt)}, cnts: {cnt}")
            peri = cv2.arcLength(cnt, True) # 计算轮廓周长
            approx = cv2.approxPolyDP(cnt, 0.005 * peri, True) # 轮廓近似
            
            # hull = cv2.convexHull(cnt, returnPoints=True) #给定二维平面上的点集，凸包就是将最外层的点连接起来构成的凸多边形，它能包含点集中所有的点
            # X = hull.squeeze().tolist()
            # if [point1[0],point1[1]] not in X: X.append([point1[0],point1[1]])
            # if [point2[0],point2[1]] not in X: X.append([point2[0],point2[1]])
            # tip_cluster = KMeans_arrow(np.array(X))
            # if [point1[0],point1[1]] in tip_cluster:
            #     start_point = point2
            #     end_point = point1
            # else:
            #     start_point = point1
            #     end_point = point2
            arrow_info.append({"start": point1, "end": point2})

        return arrow_info
    else:
        return None

def recognize_arrow_tips(img_path, bonding_boxes, save_path="", img_name="img_name"):
    image = cv2.imread(img_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    for box in bonding_boxes:
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 255), -1)
    cv2.imwrite(f"{save_path}/cover_gray_image_{img_name}.png", image)
    _, thresh_image = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite(f"{save_path}/cover_thresh_image_{img_name}.png", gray_image)
    arrow_image = get_filter_arrow_image(thresh_image)
    cv2.imwrite(f"{save_path}/get_filter_arrow_image_{img_name}.png", arrow_image)
    res_arrow_info = list()
    if arrow_image is not None:
        arrow_info = get_arrow_info(arrow_image, save_path, img_name)
        # arrow_info_image = cv2.cvtColor(arrow_image.copy(), cv2.COLOR_GRAY2BGR)
        arrow_info_image = image
        if arrow_info!=None:
            for arrow in arrow_info:
                angle = angle_beween_points(arrow["start"], arrow["end"])
                if angle>-10 and angle<10: 
                    res_arrow_info.append({"start": arrow["start"], "end": arrow["end"], "class": "arrow_line_right"})
                elif angle>80 and angle<100: 
                    res_arrow_info.append({"start": arrow["start"], "end": arrow["end"], "class": "arrow_line_down"})
                elif angle>170 and angle<-170: 
                    res_arrow_info.append({"start": arrow["start"], "end": arrow["end"], "class": "arrow_line_left"})
                else:
                    res_arrow_info.append({"start": arrow["start"], "end": arrow["end"], "class": "arrow_line_up"})
                
                cv2.line(arrow_info_image, arrow["start"], arrow["end"], (0, 255, 255), 1)

                cv2.circle(arrow_info_image, arrow["start"], 2, (178, 34, 34), 3)
                cv2.circle(arrow_info_image, arrow["end"], 2, (255, 0, 0), 3)
                cv2.putText(arrow_info_image, f"end:{int(angle)} ",
                            arrow["end"], cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 255), 1)
                cv2.putText(arrow_info_image, "start",
                            arrow["start"], cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 255), 1)
            
            cv2.imwrite(f"{save_path}/arrow_info_merge_{img_name}.png", arrow_info_image)
        return res_arrow_info
    else:
        return None
    
if __name__ == "__main__":
    image = cv2.imread("../../images/11.jpg")

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh_image = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow("thresh_image", thresh_image)

    arrow_image = get_filter_arrow_image(thresh_image)
    if arrow_image is not None:
        # cv2.imshow("arrow_image", arrow_image)
        # cv2.imwrite("arrow_image.png", arrow_image)

        arrow_info = get_arrow_info(arrow_image, save_path="../../results/", img_name="test_arrow")
        arrow_info_image = cv2.cvtColor(arrow_image.copy(), cv2.COLOR_GRAY2BGR)
        for arrow in arrow_info:
            angle = angle_beween_points(arrow["start"], arrow["end"])
            lenght = get_length(arrow["start"], arrow["end"])

            cv2.line(arrow_info_image, arrow["start"], arrow["end"], (0, 255, 255), 1)

            cv2.circle(arrow_info_image, arrow["start"], 2, (178, 34, 34), 3)
            cv2.circle(arrow_info_image, arrow["end"], 2, (255, 0, 0), 3)
            cv2.putText(arrow_info_image, f"end:{angle} ",
                        arrow["end"], cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 255), 1)
            cv2.putText(arrow_info_image, "start",
                        arrow["start"], cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 255), 1)
        
        cv2.imwrite("arrow_info_image34.png", arrow_info_image)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
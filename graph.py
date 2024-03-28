import math

import cv2
import copy
from node import Node


class Graph(object):

    def __init__(self, text_nodes, shape_nodes, arrow_nodes):
        self.text_nodes = text_nodes
        self.shape_nodes = shape_nodes
        self.arrow_nodes = arrow_nodes
        self.nodes = None
        self.adj_list = None
        self.visited_list = None
        self.first_state = -1

    def __exist_character(self, cordA, cordB):
        image = self.image
        xmin_A, xmax_A, ymin_A, ymax_A = cordA
        xmin_B, xmax_B, ymin_B, ymax_B = cordB
        for y in range(ymin_A, ymax_A):
            for x in range(xmin_A, xmax_A):
                if ((y < ymin_B) and (y > ymax_B) and (x < xmin_B) and (x > xmax_B)):
                    if (image[y, x] == 255):
                        return True
        return False

    def __is_collapse(self, A, B):
        """ This function return if exist a interseccion in two nodes.
        """
        # [xmin,xmax,ymin,ymax]
        if (type(A) is list):
            coordinateA = A
        else:
            coordinateA = A.get_coordinate()
        coordinateB = B.get_coordinate()
        x = max(coordinateA[0], coordinateB[0])
        y = max(coordinateA[2], coordinateB[2])
        w = min(coordinateA[1], coordinateB[1]) - x
        h = min(coordinateA[3], coordinateB[3]) - y
        if w < 0 or h < 0:
            return False

        return True

    def collapse_nodes_arrow(self):
        self.__collapse_nodes()
        
    def __collapse_nodes(self):
        """
        Check the text nodes that are inside of a shape node
        If a text node are inside or near of a shape node the value of the text
        set de value in shape node
        *check if exist more than one overlaping text nodes if is the case calculate the distance and the less distance it will be the true text in te shape node.
        """
        text_nodes = self.text_nodes
        shape_nodes = self.shape_nodes
        # check if exist characters or thers text nodes near to collapse
        # 将文字合并到图形中
        nodes_to_delate = []
        collapse_list = [None]*len(text_nodes)
        for i in range(len(shape_nodes)):
            if "arrow" in shape_nodes[i].get_class(): continue
            for j in range(len(text_nodes)):
                if (self.__is_collapse(shape_nodes[i], text_nodes[j])):
                    # Check if exist more than one
                    # Set the value of the text of the text_node that are inside of it
                    if (collapse_list[j] == None):
                        shape_nodes[i].add_text(text_nodes[j].get_text())
                        nodes_to_delate.append(text_nodes[j])
                        collapse_list[j] = i
                        # break
                    else:
                        if (self.__calculate_distance(shape_nodes[i], text_nodes[j]) < self.__calculate_distance(text_nodes[j], shape_nodes[collapse_list[j]])):
                            # shape_nodes[collapse_list[j]].set_text(None)
                            shape_nodes[collapse_list[j]].remove_text(text_nodes[j].get_text())
                            shape_nodes[i].add_text(text_nodes[j].get_text())
                            collapse_list[j] == i
                            # break
        # Delate all the nodes that are inside a shape_nodes
        for i in nodes_to_delate:
            text_nodes.remove(i)
        self.nodes = shape_nodes
        # 构造边
        arrow_nodes = self.arrow_nodes
        merged_arrow_nodes = list()
        merged_shape_ids = list()
        for i, arr_node in enumerate(arrow_nodes):
            start_min_distance, end_min_distance = float("inf"), float("inf")
            start_min_node, end_min_node = None, None
            for shape in shape_nodes:
                shape_coor = shape.get_coordinate()
                # print(f"get_class: {arr_node.get_class()}")
                if arr_node.get_class()=="arrow_line_down":
                    start_tmp_distance = self.calculate_point_distance(arr_node.get_start_point(), [(shape_coor[0]+shape_coor[1])//2, shape_coor[3]])
                    end_tmp_distance = self.calculate_point_distance(arr_node.get_end_point(), [(shape_coor[0]+shape_coor[1])//2, shape_coor[2]])
                    source, target = "bottom", "top"
                elif arr_node.get_class()=="arrow_line_up":
                    start_tmp_distance = self.calculate_point_distance(arr_node.get_start_point(), [(shape_coor[0]+shape_coor[1])//2, shape_coor[2]])
                    end_tmp_distance = self.calculate_point_distance(arr_node.get_end_point(), [(shape_coor[0]+shape_coor[1])//2, shape_coor[3]])
                    source, target = "top", "bottom"
                elif arr_node.get_class()=="arrow_line_right":
                    start_tmp_distance = self.calculate_point_distance(arr_node.get_start_point(), [shape_coor[1], (shape_coor[2]+shape_coor[3])//2])
                    end_tmp_distance = self.calculate_point_distance(arr_node.get_end_point(), [shape_coor[0], (shape_coor[2]+shape_coor[3])//2])
                    source, target = "right", "left"
                else:
                    start_tmp_distance = self.calculate_point_distance(arr_node.get_start_point(), [shape_coor[0], (shape_coor[2]+shape_coor[3])//2])
                    end_tmp_distance = self.calculate_point_distance(arr_node.get_end_point(), [shape_coor[1], (shape_coor[2]+shape_coor[3])//2])
                    source, target = "left", "right"
                    
                if start_tmp_distance<start_min_distance:
                    start_min_distance = start_tmp_distance
                    start_min_node = shape.get_idx()
                if end_tmp_distance<end_min_distance:
                    end_min_distance = end_tmp_distance
                    end_min_node = shape.get_idx()
            if start_min_node == end_min_node: continue
            if [start_min_node, end_min_node] in merged_shape_ids: continue
            merged_shape_ids.append([start_min_node, end_min_node])
            arr_node.set_start_end_node(start_min_node, end_min_node)
            arr_node.set_source_target(source, target)
            merged_arrow_nodes.append(copy.deepcopy(arr_node))
        
        self.arrow_nodes = merged_arrow_nodes
        
        return shape_nodes

    def __calculate_distance(self, A, B):
        """Calculate distance between two rectangles
        1)First propuest is:
            - Calculate the center point of the two nodes.
            - Calculate the distance between the two center points.
        """

        if (type(A) is list):
            cx1 = A[0]
            cy1 = A[1]
        else:
            coordinateA = A.get_coordinate()
            cx1 = int((coordinateA[0] + coordinateA[1]) / 2)
            cy1 = int((coordinateA[2] + coordinateA[3]) / 2)
        coordinateB = B.get_coordinate()
        cx2 = int((coordinateB[0] + coordinateB[1]) / 2)
        cy2 = int((coordinateB[2] + coordinateB[3]) / 2)

        return math.sqrt(math.pow(cx1 - cx2, 2) + math.pow(cy1-cy2, 2))

    def calculate_point_distance(self, A, B):
        # print(f"A:{A}, B:{B}")
        return math.sqrt(math.pow(A[0] - B[0], 2) + math.pow(A[1]-B[1], 2))
    
    def find_first_state(self):
        # return 0
        # edges = list(filter(lambda x: "arrow" in x.get_class(), self.nodes))
        # 找y轴最小的点
        min_y = float('inf')
        min_node_id = -1
        for i in range(len(self.nodes)):
            if "arrow" in self.nodes[i].get_class(): continue
            if self.nodes[i].get_coordinate()[2] <min_y:
                min_y = self.nodes[i].get_coordinate()[2]
                min_node_id = i
            # if (node.get_class() == "start_end" and node.get_text().lower() == "inicio"):
            #     return self.nodes.index(node)
        if min_node_id != -1: self.nodes[min_node_id].set_class('start_end')
        return min_node_id

    def __is_any_arrow(self, node):
        return node.get_class().split('_')[0] == "arrow"

    def __is_graph_visited(self):
        return (sum(x == 1 for x in self.visited_list) == len(self.visited_list))

    def __can_visit(self, previous_node, node_index):
        if (self.nodes[node_index].get_class() == "decision"):
            return self.visited_list[node_index] <= 1 and not (previous_node in self.adj_list[node_index])
        elif (self.nodes[node_index].get_class() == "start_end" and self.nodes[node_index].get_text().lower() == "fin"):
            return not (previous_node in self.adj_list[node_index])
        else:
            return self.visited_list[node_index] == 0 and not (previous_node in self.adj_list[node_index])

    def __find_next(self, node_index):
        if (not (self.__is_graph_visited())):
            # calculate the distance with another nodes
            distances = []
            nodes_prompter = []
            min_distance = float('inf')
            min_node = None
            to_compare = None
            # check only with the posibles
            # if is start end:start
            if self.nodes[node_index].get_class() == "start_end": # and self.nodes[node_index].get_text().lower() == "inicio")
                for i in range(len(self.nodes)):
                    if (node_index != i and self.__can_visit(node_index, i)):
                        distance = self.__calculate_distance(
                            self.nodes[node_index], self.nodes[i])
                        if (distance < min_distance):
                            min_distance = distance
                            min_node = i
                # min_node is the most near node
                if (self.__is_any_arrow(self.nodes[min_node])):
                    # add the adyacency
                    self.adj_list[node_index].append(min_node)
                    self.visited_list[node_index] += 1
                    return self.__find_next(min_node)
                else:
                    return "NV"
            # if is any arrow
            # check the type of arrow to predict better
            elif (self.__is_any_arrow(self.nodes[node_index])):
                # find the point of check the distance between the other node
                point_to_compare = None
                ac = self.nodes[node_index].get_coordinate()
                if (self.nodes[node_index].get_class() == "arrow_line_down"):
                    point_to_compare = [(ac[0] + ac[1])/2, ac[3]]
                elif (self.nodes[node_index].get_class() == "arrow_line_left"):
                    point_to_compare = [ac[0], (ac[2] + ac[3])/2]
                elif (self.nodes[node_index].get_class() == "arrow_line_right"):
                    point_to_compare = [ac[1], (ac[2] + ac[3])/2]
                elif (self.nodes[node_index].get_class() == "arrow_line_up"):
                    point_to_compare = [(ac[0] + ac[1])/2, ac[2]]
                for i in range(len(self.nodes)):
                    if (node_index != i and self.__can_visit(node_index, i)):
                        # calculate the distance
                        distance = self.__calculate_distance(
                            point_to_compare, self.nodes[i])
                        if (distance < min_distance):
                            min_distance = distance
                            min_node = i
                self.adj_list[node_index].append(min_node)
                self.visited_list[node_index] += 1
                return self.__find_next(min_node)
            # if is process,print, or scan
            elif (self.nodes[node_index].get_class() == "print" or self.nodes[node_index].get_class() == "process" or self.nodes[node_index].get_class() == "scan"):
                for i in range(len(self.nodes)):
                    if (node_index != i and self.__can_visit(node_index, i)):
                        distance = self.__calculate_distance(
                            self.nodes[node_index], self.nodes[i])
                        if (distance < min_distance):
                            min_distance = distance
                            min_node = i
                # min_node is the most near node
                if (self.__is_any_arrow(self.nodes[min_node])):
                    # add the adyacency
                    self.adj_list[node_index].append(min_node)
                    self.visited_list[node_index] += 1
                    return self.__find_next(min_node)
                else:
                    return "NV"
            # if is decision
            elif (self.nodes[node_index].get_class() == "decision"):
                if (self.visited_list[node_index] == 0):
                    for i in range(len(self.nodes)):
                        if (node_index != i and self.__can_visit(node_index, i) and self.nodes[i].get_text() != None):
                            distance = self.__calculate_distance(
                                self.nodes[node_index], self.nodes[i])
                            distances.append(distance)
                            nodes_prompter.append(i)
                    node_distance = list(zip(distances, nodes_prompter))
                    node_distance = sorted(node_distance)
                    to_check = node_distance[0:2]
                    if (self.__is_any_arrow(self.nodes[to_check[0][1]]) and self.__is_any_arrow(self.nodes[to_check[1][1]])):
                        self.adj_list[node_index].append(to_check[0][1])
                        self.adj_list[node_index].append(to_check[1][1])
                        self.visited_list[node_index] += 1
                        a = self.__find_next(to_check[0][1])
                        b = self.__find_next(to_check[1][1])
                    else:
                        return str(node_index)+"NV"

    def generate_graph(self):
        """ Generate the adyacency list of the nodes starting of the relationship of
        the nodes. Decision shape is a special case, because can have two connectors
        *Check the cases arrow-rectangle/arrow line.
        """
        self.nodes = self.__collapse_nodes()
        # print(f"\n __collapse_nodes nodes: {list(enumerate(self.nodes))}\n")
        self.adj_list = {key: [] for key in range(len(self.nodes))}
        self.visited_list = [0]*len(self.nodes)
        self.first_state = self.find_first_state()
        # print(f"first state: {self.first_state}\n")
        if (self.first_state == -1):
            return "self.first_state == -1 Not valid init"
        if (self.__find_next(self.first_state) == "NV"):
            # print(self.adj_list)
            return "NVNVNV Not valid"
        # print(f"\nadj_list: {self.adj_list}\n")
        return self.adj_list

    def get_adyacency_list(self):
        return self.adj_list

    def get_nodes(self):
        return self.nodes
        
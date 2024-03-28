class Node(object):
    """Node represents a simple element of the flowchart."""

    def __init__(self, idx, coordinate, text="", class_shape=None, image_path=None):
        self.id = idx
        self.coordinate = coordinate
        self.text = text
        self.class_shape = class_shape
        self.image_path = image_path

    def collapse(self, node):
        xmin_A,xmax_A,ymin_A,ymax_A = self.get_coordinate()
        xmin_B,xmax_B,ymin_B,ymax_B = node.get_coordinate()
        n_xmin = min(xmin_A,xmin_B)
        n_xmax = max(xmax_A,xmax_B)
        n_ymin = min(ymin_A,ymin_B)
        n_ymax = max(ymax_A,ymax_B)
        self.set_coordinate([n_xmin,n_xmax,n_ymin,n_ymax])

    def get_idx(self):
        return self.id
    
    def get_coordinate(self):
        """Returns a tuple (x1, x2, y1, y2)."""
        return self.coordinate

    def get_center_point_top(self):
        return self.coordinate[2]*1.5
        return (self.coordinate[2] + self.coordinate[3])*1.5 // 2
    
    def get_center_point_left(self):
        return self.coordinate[0]*1.5
        return (self.coordinate[0] + self.coordinate[1])*1.5 // 2
    
    def get_node_size(self):
        return {
                "width": f"{abs(self.coordinate[1] - self.coordinate[0])}px", 
                "height": f"{abs(self.coordinate[3] - self.coordinate[2])}px",
                "rows": len(self.text.split("\n"))
                }
    
    def get_text(self):
        return self.text

    def set_class(self, class_type):
        self.class_shape = class_type
        
    def get_class(self):
        """Return class of shape."""
        return self.class_shape

    def set_coordinate(self,coordinate):
        self.coordinate = coordinate

    def set_text(self,text):
        self.text = text

    def add_text(self,text):
        if self.text == None or self.text=="": self.text = text
        else: self.text += "\n" + text
    
    def remove_text(self,text):
        self.text.replace(f"\n{text}", '')
        
    def set_class(self,class_shape):
        self.class_shape = class_shape

    def get_type(self):
        """Return type of node (None, text, shape or connector)."""

        if(self.class_shape == None and self.text == None):
            return None
        if(self.class_shape == None):
            return 'text'
        if(self.text == None):
            return 'connector'
        return 'shape'

    def get_image_path(self):
        """Return the path of the cropped node (only rectangle arrows)."""

        return self.image_path

    def __str__(self):
        return "Node(coord:"+str(self.coordinate)+",class:"+str(self.class_shape)+",text:"+str(self.text)+")"

    def __repr__(self):
        return "Node(coord:"+str(self.coordinate)+",class:"+str(self.class_shape)+",text:"+str(self.text)+")"

class Arrow(object):
    def __init__(self, start_point, end_point, edge_class ) -> None:
        self.start_point = start_point
        self.end_point = end_point
        self.edge_label = ""
        self.start_node = ""
        self.end_node = ""
        self.edge_class = edge_class
        self.source = "top"
        self.target = "bottom"
        # arrow_line_right arrow_line_down  arrow_line_left arrow_line_up
    
    def get_label(self):
        return self.edge_label
    
    def set_label(self, label):
        self.edge_label = label
        
    def get_class(self):
        return self.edge_class
    
    def set_class(self, e_class):
        self.edge_class = e_class
    
    def set_start_end_node(self, s_node, e_node):
        self.start_node = s_node
        self.end_node = e_node
    
    def get_start_point(self):
        return self.start_point
    
    def get_end_point(self):
        return self.end_point
    
    def get_start_node(self):
        return self.start_node
    
    def get_end_node(self):
        return self.end_node
    
    def set_source_target(self, s, t):
        self.source = s 
        self.target = t
    
    def __str__(self):
        return f"Arrow(start:{str(self.start_point)}, end:{str(self.end_node)}, class:{str(self.edge_class)}, text:{str(self.edge_label)})"

    def __repr__(self):
        return f"Arrow(start:{str(self.start_point)}, end:{str(self.end_node)}, class:{str(self.edge_class)}, text:{str(self.edge_label)})"

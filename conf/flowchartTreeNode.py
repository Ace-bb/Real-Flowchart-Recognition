class FlowChartTreeNode:
    def __init__(self, nid, pid, content, label=None, source=None, isAction=False) -> None:
        self.id = int(nid)
        self.parent = int(pid)
        self.content = content.replace('\n', '')
        self.childs = list()
        self.label = label
        self.source =  source
        self.isAction = isAction
        self.augment_content = ""
        pass

    def __str__(self) -> str:
        return f"id:{self.id}, parent:{self.parent}, content:{self.content}, label:{self.label}"
    
    def is_empty(self):
        return len(self.childs)==0
    
    def to_dict(self) -> dict:
        res_dic = {
            "id": self.id,
            "parent": self.parent,
            "content": self.content,
            "augment_content": self.augment_content,
            "isAction": self.is_empty(),
            "label": "",
            "source": ""
        }
        if self.label!=None or self.label!="": res_dic['label'] = self.label
        if self.source!=None or self.source!="": res_dic['source'] = self.source
        return res_dic
    
    def set_augment_content(self, c):
        self.augment_content = c
    

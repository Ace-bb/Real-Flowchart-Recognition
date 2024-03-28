from conf.flowchartTreeNode import FlowChartTreeNode as Node

flowchart_tree_data = {
    "成人流行性感冒诊疗规范急诊专家共识": [
        Node(nid=1, pid=0, content="临床表现任何2条 ①发热，体温>=37.8℃； ②新发呼吸系统症状或加重，包括但不限于咳嗽，喉咙痛，鼻塞或流鼻涕等；③新发全身症状或加重，包括但不限于肌痛，头痛，出汗，发冷或疲劳乏力等。"),
        Node(nid=2, pid=1, content="必要时进行新型冠状病毒筛查"),
        Node(nid=3, pid=1, content="询问流感流行病学病史 ①流感季节发病；②发病前7天内曾到过流感暴发疫区；③有与确诊或疑似流感患者密切接触史；④与禽类动物接触史"),
        Node(nid=4, pid=3, content="疑似诊断", label="有"),
        Node(nid=5, pid=3, content="完善辅助检查（满足1条）①外周白细胞和淋巴细胞计数正常或减少；②肺部X线片或CT疑似病毒性肺炎表现。", label="无"),
        Node(nid=6, pid=5, content="其他疾病筛查", label="无"),
        Node(nid=7, pid=4, content="排除其他流感样表现的疾病"),
        Node(nid=8, pid=4, content="流感病原学急诊筛查 ①抗原；②快速核算；③流感病毒分离；④抗体水平检测"),
        Node(nid=9, pid=8, content="其他疾病筛查", label="阴性"),
        Node(nid=10, pid=8, content="确诊", label="任意一个阳性"),
        Node(nid=11, pid=7, content="临床诊断流感")
    ]
}
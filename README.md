# Real-Flowchart-Recognition
一个流程图识别的流水线框架，基于**Faster R-CNN**检测流程图中的基本图形和连接线，使用**PaddleOCR**识别文字信息，然后将整合**图形**，**连接线**和**文字**，使用节点和边来表示流程图，最终输出为JSON格式。

## Overview
目前流程图识别的文章或开源项目，大都以检测出流程图中的图形的bounding-box为主，并没有将检测出的图形，连接线和文字信息重新组合成流程图。
因此，本项目开源了一个从图形检测、文字识别到流程图重组的流水线框架。并且提供了Faster R-CNN训练数据自动化构建工具，能够构建出指定大小的训练数据集，并且用来训练Faster R-CNN。
训练完成的模型用来检测流程图中的基本图形，检测和识别文字部分主要使用了PaddleOCR的PP-OCRv4模型，基本图形和文字都识别完成之后，再根据bbox将文字融合进图形中，形成一个节点。
最后根据检测出的连接关系，重构流程图，最终构造成有向图的形式。并保存为JSON格式。

## Set up
1. 创建一个虚拟环境，推荐使用Anaconda进行创建
2. 克隆本仓库：`git clone https://github.com/Ace-bb/Real-Flowchart-Recognition.git`
3. 激活创建的虚拟环境
   1. 安装Pytoch：`conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia`
   2. 安装PaddlePaddle：`conda install paddlepaddle-gpu==2.6.1 cudatoolkit=11.7 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ -c conda-forge`
   3. 安装PaddleOCR：`pip install "paddleocr>=2.0.1" # 推荐使用2.0.1+版本`
   4. 安装剩下的包：`pip install -r requirements.txt`
4. 下载模型
   1. 下载Faster R-CNN backbone
        进入FasterRCNN中的backbone文件夹并下载模型：

```bash
cd FasterRCNN/backbone
wget https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
mv fasterrcnn_resnet50_fpn_coco-258fb6c6.pth fasterrcnn_resnet50_fpn_coco.pth
wget https://download.pytorch.org/models/resnet50-19c8e357.pth
mv resnet50-19c8e357.pth resnet50.pth
```
   2. 下载训练后的流程图检测模型参数，训练后的模型参数保存在`FasterRCNN/save_weights`中
```bash
cd FasterRCNN/save_weights
wget https://drive.google.com/file/d/17mKO2BUrEiIL32BBhx6M7kVGCO_f7CO1/view?usp=drive_link
tar xf flowchart-recognition-based-fasterrcnn.tar.gz
```
   3. 下载PaddleOCR模型
```bash
mkdir models & cd models
mkdir paddleocr & cd paddleocr
wget https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar & tar xf ch_PP-OCRv4_det_infer.tar
wget https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar & tar xf ch_PP-OCRv4_rec_infer.tar
```
    PaddleOCR的其他模型可以在这里下载：[PP-OCR系列模型列表](https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_ch/models_list.md)


## Usage
1. 激活Conda环境
2. 移动到项目的根目录，在`images`目录下创建一个文件夹`folder_name`(自己命名),将需要识别的图片存放到`images/folder_name`中
3. 运行`main.py`文件：`python main.py`
4. 识别的结果将保存到`results/folder_name`目录中，同名文件夹为识别过程中的输出，可以看到各部分的识别效果。识别结果的JSON格式，保存在`file_name.json`中。

## Examples of th results
[example1](results/f1/f1/draw-node-edge.png)
[example2](results/flowchart/2/draw-node-edge.png)

## Coming soon
1. 提升连接线的识别效果
2. 支持更多流程图基本图形的识别
3. 支持导出多种格式，markdown格式等
4. 推出识别结果在线编辑工具
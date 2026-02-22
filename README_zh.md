## YOLO-Globe

本项目是通过用虚幻引擎 5 (UE5) 构建了一套自动化合成数据生成的 Pipeline（合成地球仪图像+标注）；使用 YOLO26 进行目标检测；利用 Sim-to-Real（仿真到现实）技术验证模型效果；使用 LoRA 微调解决小目标（小国家）和边缘畸变问题。

### 1.数据准备

数据的来源分为实拍和虚拟合成。实拍部分是使用手机拍摄的，因此略过；而虚拟部分则是使用UE5通过场景变换和取景器组件截图生成的。相关项目可从网盘下载，UE5版本为5.4。

链接: https://pan.baidu.com/s/1ZeXob0uif4bUi0tH5CXgXA?pwd=sbqs 提取码: sbqs 

进入场景运行后，按 __P__ 键即可开始自动截图，生成的图像结果如图所示。

![synthetic_samples](figures\ue_synthetic_samples.png)

生成的数据包含了图像和标注，标注需要通过python脚本转换为 __ultralytics__ 库能够处理的格式。处理的脚本为：__make_yolo_dataset.py__

真实和虚拟的数据集的放置路径如下，这一点从 __ultralytics__ 的官方文档中也能找到。
![dataset_placement](figures\dataset_placement.png)

图片和标注的样本示例如图所示：
![dataset_image](figures\dataset_image.png)
![dataset_label](figures\dataset_label.png)

数据增强的背景使用了 __COCO2017__ 数据集：

下载地址如下：

http://images.cocodataset.org/zips/train2017.zip
http://images.cocodataset.org/zips/test2017.zip
http://images.cocodataset.org/zips/val2017.zip

### 2.模型训练

YOLO系列模型训练依赖于 __ultralytics__ 库，首先下载YOLO26的权重到 __checkpoints__ 文件夹中。默认使用的模型权重是 __yolo26m.pt__。

配置好数据和权重后，使用 __train.py__ 即可训练。

这个脚本同样还包含了基于 __peft__ 库的微调实现，具体是利用LoRA对于YOLO26中的卷积层进行旁路微调。

结果存放于 __run__ 文件夹中。

### 3.其他

真实数据的标注使用的工具是 __vott__。

论文投稿中，权重、数据等更详细的信息会在中稿后进一步发布。
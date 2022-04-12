通过 yolov5 + openpose实现摔倒检测

需要用的模型文件可以在网盘下载

链接：https://pan.baidu.com/s/1oL5wmEdZW_3KehArrK9PNw
提取码：3su3
--来自百度网盘超级会员V4的分享

运行runOpenpose.py  
只跑了open pose 可以获得人体的关键点图，用于后续的.jit模型训练 
人体的关键点图会保存在data/test中
pose.py中draw方法的最下面可以控制保存关键点图的位置

运行detect.py  
1.先通过yolo检测图片中的人，遍历将人的box框传给openpose
2.open pose通过box框中的图片进行关键点检测以及人体的姿态检测
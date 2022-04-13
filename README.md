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
会先进行yolo目标检测，检测到人后，detect.py 169 行，加了根据框框的宽高比的判断，后续可自行修改
根据坐标将人的图片扣出给openpose  进行姿态检测  
runOpenpose.py 159行 加了一些限制，后续可自行修改


如果想要检测其他姿势：
1.收集图片，跑runOpenpose.py 文件获得人体的关键点图
2.对人体的关键点图根据自己想要的进行分类放在data/train 和 data/test
3.跑 action_detect/train.py
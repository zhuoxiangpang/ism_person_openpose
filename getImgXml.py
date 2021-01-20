import os

path = 'C:\\zqr\\project\\yolov5\\data\\images'
f1 = open('ImageXML.txt', 'w')
fileList = []
files = os.listdir(path)
for f in files:
    if (os.path.isfile(path + '/' + f)):
        # 添加文件
        fileList.append(f)
for fl in fileList:
    # 打印文件
    out_path =  fl.split(".")[0]
    print(out_path)
    f1.write(out_path + '\n')
f1.close()


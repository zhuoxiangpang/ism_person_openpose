import cv2
import xml.etree.ElementTree as ET
import os


def convert_annotation(image_id):

    in_file = open(r'./Annotations/%s.xml' % (image_id), 'rb')  # 读取xml文件路径
    img = cv2.imread('./images/%s.jpg' %(image_id))
    out_file = open('./labels/%s.txt' % (image_id), 'w')  # 需要保存的txt格式文件路径
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        # cls = obj.find('name').text
        # if cls not in classes:  # 检索xml中的缺陷名称
        #     continue
        # cls_id = classes.index(cls)

        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        print(b)
        cv2.rectangle(img, (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text)), (int(xmlbox.find('xmax').text),int(xmlbox.find('ymax').text)), (0, 255, 0), 2)

        # bb = convert((w, h), b)
        # out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    # img = cv2.resize(img, (640, 480))
    cv2.imwrite('./test/%s.jpg' % (image_id), img)
    # cv2.waitKey(500)

file = './label/'
file1 = 'Annotations/'
num = 0
# for i in os.listdir(file):
#     num += 1
#     if num <=75:
#         continue
#     print(num, '***', i)
#     # if 'png' in i:
#     #     os.rename(file + i, file + i.replace('png', 'jpg'))
#     #     print(i)
#     try:
#         convert_annotation(i.split('.')[0])
#     except:
#         nana.append(i)

for i in os.listdir(file):
    with open(file + i) as f:
        line = f.read()
        if len(line) <=10:
            print(i)
    # flag = 0
    # for j in os.listdir(file1):
    #     if i.split('.')[0] ==j.split('.')[0]:
    #         flag = 1
    # if flag == 0:
    #     print(file + i)
    #     os.remove(file + i)
    # try:
    #     os.remove('Annotations/' + i.replace('jpg', 'xml'))
    # except:
    #     print(i)
import argparse
import time
from pathlib import Path
from torch import from_numpy, jit

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import os
import datetime
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

import runOpenpose


def detect(save_img=False):
    global ip
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # 加载摔倒检测的模型
    print("加载摔倒检测的模型开始")
    net = jit.load(r'.\action_detect\checkPoint\openpose.jit')
    action_net = jit.load(r'.\action_detect\checkPoint\action.jit')
    print("加载摔倒检测的模型结束")
    # Initialize
    set_logging()
    # 获取设备
    device = select_device(opt.device)
    # 如果设备为gpu，使用Float16
    half = device.type != 'cpu'  # half precision only supported on CUDA
    # Load model
    # 加载Float32模型，确保用户设定的输入图片分辨率能整除32（如果不能则调整为能整除并删除）
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    # 设置Float16
    if half:
        model.half()  # to FP16

    # Set Dataloader
    # 通过不同的输入源来设置不同的数据加载方式
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        # 如果检测视频的时候想显示出来，可以再这里加一行 view_img = True
        view_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    # 获取类别的名字
    names = model.module.names if hasattr(model, 'module') else model.names
    # 设置画框的颜色
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    t0 = time.time()
    # 进行一次向前推理，测试程序是否正常
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    """
        path 图片/视频路径
        img 进行resize+pad之后的图片
        img0 原size图片
        cap 当读取图片时为None，读取视频时为视频源
        """
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        # 图片也设置为Float16
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        # 没有batch_size 的话则在最前面增加一个轴
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        """
               前向传播 返回pred的shape是(1, num_boxes, 5+num_class)
               h,w为传入网络图片的长和宽，注意dataset在检测时使用了矩形推理，所以这里h不一定等于w
               num_boxes = h/32 * w/32 + h/16 * w/16 + h/8 * w/8
               pred[..., 0:4]为预测框坐标
               预测框坐标为xywh(中心点+宽长)格式
               pred[..., 4]为objectness置信度
               pred[..., 5:-1]为分类结果
               """
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        """
               pred:前向传播的输出
               conf_thres:置信度阈值
               iou_thres:iou阈值
               classes:是否只保留特定的类别
               agnostic:进行nms是否也去除不同类别之间的框
               经过nms之后，预测框格式：xywh-->xyxy(左上角右下角)
               pred是一个列表list[torch.tensor]，长度为batch_size
               每一个torch.tensor的shape为(num_boxes, 6),内容为box+conf+cls
               """
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        # 添加二次分类，默认不适用
        # if classify:
        #     pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        # 对每一张图片作处理
        for i, det in enumerate(pred):  # detections per image
            # 如果输入源是webcam 则batch_size 不为1，取出dataset中的一张图片
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            # print(det)

            boxList = []  # 框的一个list 给openpose 使用
            p = Path(p)  # to Path
            # 设置保存图片/视频的路径
            save_path = str(save_dir / p.name)  # img.jpg
            # 设置保存框最薄txt文件的路径
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            # 设置打印信息(图片长宽)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                # 调整预测框的坐标：基于resize+pad的图片的坐标 -->基于原size图片的坐标
                # 此时坐标格式为xyxy
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                # 打印检测到的类别数量
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f'{n} {names[int(c)]}s, '  # add to string

                # Write results
                # 保存预测结果
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        # 将xyxy（左上角+右下角）格式转为xywh（中心点+宽长）格式，并除上w,h做归一化，转化为列表再保存
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    # 在原图上画框
                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                        boxList.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])

            print(boxList)
            for box in boxList:
                c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                x = c2[0] - c1[0]
                y = c2[1] - c1[1]
                if x / y >= 0.8:  # 比例>0.8 可能会是摔倒
                    print('进行人体姿态检测')
                    runOpenpose.run_demo(net, action_net, im0, 256, True, boxList)  # 人体姿态检测 将图片和yolov5检测人体的框也传给openpose
                    break
            # Print time (inference + NMS)
            # 打印向前传播+nms时间
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            if save_img:
                if dataset.mode == 'image':
                    imageName = str(time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))) + ".jpg"
                    imagePath = r"/photo/ism_uniform/" + str(ip) + "/" + str(
                        datetime.datetime.now().strftime("%Y-%m-%d")) + "/"
                    if not os.path.exists(r"/photo/ism_uniform/" + str(ip) + "/"):
                        os.makedirs(r"/photo/ism_uniform/" + str(ip) + "/")
                    if not os.path.exists(r"/photo/ism_uniform/" + str(ip) + "/" + "/" + str(
                            datetime.datetime.now().strftime("%Y-%m-%d"))):
                        os.makedirs(r"/photo/ism_uniform/" + str(ip) + "/" + "/" + str(
                            datetime.datetime.now().strftime("%Y-%m-%d")))
                    cv2.imwrite(imagePath + imageName, im0)
                else:  # 'video'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)
                print('保存图片')

    # 打印总时间
    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 选用训练的权重，可用根目录下的yolov5s.pt，也可用runs/train/exp/weights/best.pt
    parser.add_argument('--weights', type=str, default='models/yolov5s.pt', help='model.pt path(s)')
    # 检测数据，可以是图片/视频路径，也可以是'0'(电脑自带摄像头),也可以是rtsp等视频流
    parser.add_argument('--source', type=str, default='D:\\project\\ism_person_openpose\\data\\pics',
                        help='source')  # file/folder, 0 for webcam
    # 网络输入图片大小
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    # 置信度阈值，检测到的对象属于特定类（狗，猫，香蕉，汽车等）的概率
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    # 做nms的iou阈值
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    # 检测的设备，cpu；0(表示一个gpu设备cuda:0)；0,1,2,3(多个gpu设备)。值为空时，训练时默认使用计算机自带的显卡或CPU
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # 是否展示检测之后的图片/视频，默认False
    parser.add_argument('--view-img', action='store_true', help='display results')
    # 是否将检测的框坐标以txt文件形式保存，默认False
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    # 是否将检测的labels以txt文件形式保存，默认False
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    # 设置只保留某一部分类别，如0或者0 2 3
    parser.add_argument('--classes', nargs='+', default=0, type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    # 进行nms是否也去除不同类别之间的框，默认False
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    # 推理的时候进行多尺度，翻转等操作(TTA)推理
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    # 如果为True，则对所有模型进行strip_optimizer操作，去除pt文件中的优化器等信息，默认为False
    parser.add_argument('--update', action='store_true', help='update all models')
    # 检测结果所存放的路径，默认为runs/detect
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    # 检测结果所在文件夹的名称，默认为exp
    parser.add_argument('--name', default='exp', help='save results to project/name')
    # 若现有的project/name存在，则不进行递增
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect()

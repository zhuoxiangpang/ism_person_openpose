import torch
from action_detect.data import *
from action_detect.net import *
from torch.utils.data import DataLoader
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import time

DEVICE = "cuda:0"

class Train:
    def __init__(self,root):

        self.summmaryWriter = SummaryWriter("./logs")

        # 加载训练数据
        self.train_dataset = PoseDataSet(root,True)
        self.train_dataLoader = DataLoader(self.train_dataset,batch_size=100,shuffle=True)

        # 加载测试数据
        self.test_dataset = PoseDataSet(root,False)
        self.test_dataLoader = DataLoader(self.test_dataset,batch_size=100,shuffle=True)

        #创建模型
        # self.net = NetV1()
        self.net = NetV2()
        #加载已训练的数据
        self.net.load_state_dict(torch.load("D:/py/openpose_lightweight/action_detect/checkPoint/action.pt"))
        self.net.to(DEVICE)  # 使用GPU进行训练


    #    定义优化器
        self.opt = optim.Adam(self.net.parameters()) #加强版梯度下降法,SGD 普通梯度下降法

    # 启动训练
    def __call__(self):
        for epoch in range(100000):
            train_sum_loss = 0
            for i, (imgs, tags) in enumerate(self.train_dataLoader):

                #训练集添加到GPU
                imgs,tags = imgs.to(DEVICE),tags.to(DEVICE)
                self.net.train() #表明在训练环境下进行

                train_y = self.net(imgs)
                loss = torch.mean((tags-train_y)**2)

                self.opt.zero_grad() #清梯度
                loss.backward()
                self.opt.step()

                train_sum_loss += loss.cpu().detach().item()

            train_avg_loss = train_sum_loss/len(self.train_dataLoader)
            # print(epoch,avg_loss)
            # sum_score = 0
            test_sum_loss = 0
            for i, (imgs, tags) in enumerate(self.test_dataLoader):
                # 测试集添加到GPU
                imgs, tags = imgs.to(DEVICE), tags.to(DEVICE)

                self.net.eval() #标明在测试环境下

                test_y = self.net(imgs)
                loss = torch.mean((tags - test_y) ** 2)
                test_sum_loss += loss.cpu().detach().item()

                predict_targs = torch.argmax(test_y,dim=1)
                label_tags = torch.argmax(tags,dim=1)
                # sum_score += torch.eq(predict_targs,label_tags).float().cpu().detach().item() #正确的是1错误的是0

            test_avg_loss = test_sum_loss / len(self.test_dataLoader)
            # score = sum_score/len(self.test_dataset)

            self.summmaryWriter.add_scalars("loss",{"train_avg_loss":train_avg_loss,"test_avg_loss":test_avg_loss},epoch)
            # self.summmaryWriter.add_scalar("score",score,epoch)
            print(epoch, train_avg_loss, test_avg_loss)

            #添加时间戳
            now_time = int(time.time())

            timeArray = time.localtime(now_time)

            str_time = time.strftime("%Y-%m-%d-%H:%M:%S", timeArray)

            torch.save(self.net.state_dict(),f"./checkPoint/action.pt") #保存训练的数据


if __name__ == '__main__':
    train = Train('C:/Users/lieweiai/Desktop/human_pose')
    train();
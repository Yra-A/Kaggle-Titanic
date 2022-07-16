# @File : Titanic.py
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pandas as pd

#训练集中的乘客年龄平均值
train_AgeMean = 0

#准备数据集
class TitanicDataset(Dataset):
    def __init__(self, filepath):
        xy = pd.read_csv(filepath)
        self.len = xy.shape[0]
        #数据集中的"Age"部分有数据缺失，这里用平均值补上去
        train_AgeMean = xy["Age"].mean()
        xy["Age"] = xy["Age"].fillna(train_AgeMean)
        #选择了6个特征 在这里维度为7，是因为Sex特征的one-hot向量有两维
        feature = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
        #将数据转化成矩阵, 对于x需要用one-hot编码
        self.x_data = torch.from_numpy(np.array(pd.get_dummies(xy[feature])))
        self.y_data = torch.from_numpy(np.array((xy["Survived"])))
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        return self.len

dataset = TitanicDataset("train.csv")

#采用Mini-Batch训练
train_loader = DataLoader(dataset = dataset, batch_size = 1, shuffle = True, num_workers = 2)

#定义模型
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(7, 4)
        self.linear2 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    #前向传播
    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        return x

    #测试函数
    def test(self, x):
        x = self.forward(x)
        #将x压缩为1维
        x = x.squeeze(-1)
        #将x从计算图中分离，去除梯度
        x = torch.detach(x)
        #对预测值四舍五入，预测最终的生存结果，并转化为int和列表形式
        y = np.int_(np.around(x)).tolist()
        return y

model = Model()

#损失函数 二分类交叉熵
criterion = torch.nn.BCELoss(reduction = "mean")
#随机梯度下降
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

def train():
    for epoch in range(100):
        for batch_idx, data in enumerate(train_loader, 0):
            inputs, labels = data
            #将数据由double转化为float
            inputs = inputs.float()
            labels = labels.float()
            #进行预测
            outputs = model(inputs)
            #将预测和标签的向量维度统一
            outputs = outputs.squeeze(-1)
            #计算损失
            loss = criterion(outputs, labels)
            print(epoch, batch_idx, loss.item())
            #先梯度清零，否则会累加
            optimizer.zero_grad()
            #反向传播，计算参数梯度值
            loss.backward()
            #更新参数
            optimizer.step()

#测试函数
def test():
    #读入数据
    xy = pd.read_csv("test.csv")
    xy["Age"] = xy["Age"].fillna(train_AgeMean)
    feature = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
    x_data = torch.from_numpy(np.array(pd.get_dummies(xy[feature])))
    x_data = x_data.float()
    #调用测试函数
    y = model.test(x_data)
    #输出最终的预测结果
    output = pd.DataFrame({'PassengerId': xy.PassengerId, 'Survived': y})
    output.to_csv("my_predic.csv", index = False)

if __name__ == "__main__":
    train()
    test()




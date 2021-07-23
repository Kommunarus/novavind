import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import os




def customloss(pred, true):

    dist = torch.cdist(pred,true)
    m,n = dist.shape
    s = torch.Tensor([0.0])
    y = dist.clone()
    if m != n:
        sch = min(n,m)
    else:
        sch = n
    dd = torch.Tensor([0]*sch)
    for i in range(sch):
        if i == torch.Tensor([sch-1]):
            # s = s + y[0]
            dd[i] = torch.min(y)
        else:
            min_in_row = torch.min(y, 1)

            minval = torch.min(min_in_row[0])
            index_min = torch.argmin(min_in_row[0])

            # s = s + min
            dd[i] = minval
            r = index_min
            c = min_in_row[1][index_min]

            if r == torch.Tensor([0]):
                y = y[1:, :]
            elif r == torch.Tensor([y.shape[0]-1]):
                y = y[:-1, :]
            else:
                y = torch.cat((y[0:r, :], y[r+1:, :]), 0)

            if c == torch.Tensor([0]):
                y = y[:, 1:]
            elif c == torch.Tensor([y.shape[1]-1]):
                y = y[:, :-1]
            else:
                y = torch.cat((y[:, 0:c], y[:, c+1:]), 1)

    return torch.max(dd)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 2)
        # nn.init.xavier_uniform_(self.fc.weight)
        with torch.no_grad():
            K = torch.Tensor([[1,0],[0,1]])
            self.fc.weight = torch.nn.Parameter(K)

    def forward(self, x):
        x = self.fc(x)
        return x


def findoptim(f1, outdict, dir):
    f1 = [[y, x] for x, y, _ in f1]
    # all_vetr = [[1300,3400],[2222,3500],[3200,3500],[4300,3700],[5600,3800],[7000,4000],
    #       [6400,3000],[5670,3000],[5000,3000],[4350,3000],[3750,2900],[3100,2900]]
    weght_v = 30
    weght_h = 60
    minx = 0
    maxx = 1
    miny = 0
    maxy = 1

    f3 = [[(weght_h - 1) * (value['x']) / (maxx - minx), (weght_v - 1) * (value['y']+0.5*value['h']) / (maxy - miny)]
             for key, value
                in outdict.items()]

    min_y_f1 = min([y for x,y in f1])
    min_y_f3 = min([y for x,y in f3])

    f3 = [[x,y + (min_y_f1-min_y_f3)] for x,y in f3]


    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

    for i, (x, y) in enumerate(f1):
        ax.scatter(x, y, marker='v', c='tab:blue')
        ax.annotate(str(i), (x, y))
    for i, (x, y) in enumerate(f3):
        ax.scatter(x, y, marker='o', c='tab:orange')
        ax.annotate(str(i), (x, y))
    fig.gca().invert_yaxis()
    fig.savefig(os.path.join(dir,'fig1.jpg'))



    net = Net()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    net.train()
    for epoch in range(2000):
        optimizer.zero_grad()
        input = torch.from_numpy(np.array([[x, y] for x, y in f1]))
        input = input.type(torch.float32)
        out = net(input)

        target = torch.from_numpy(np.array([[x, y] for x, y in f3]))
        target = target.type(torch.float32)

        lossCustom = customloss(out, target)
        # loss = criterion(out, target)
        lossCustom.backward()
        optimizer.step()
    print('{:.3f}'.format(lossCustom.item()))

    net.eval()
    input = torch.from_numpy(np.array([[x, y] for x, y in f1]))
    input = input.type(torch.float32)
    pred = net(input)

    # f3 - детектор. к нему устремляются мат точки
    dist = torch.cdist(torch.Tensor(f3),pred)
    row_ind, col_ind = linear_sum_assignment(dist.detach().numpy())
    # for x,y in zip(row_ind, col_ind):
    #     print(x,y)
    # row - точки детектора, кол - точки из списка теории
    out = pred.detach().numpy()
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    for i, (x, y) in enumerate(f3):
        ax.scatter(x, y, marker='*', c='tab:orange')
        ax.annotate(str(i), (x, y))
    for i in range(out.shape[0]):
        ax.scatter(out[i, 0], out[i, 1], marker='+', c='tab:green')
        ax.annotate(str(i), (out[i, 0], out[i, 1]))
    fig.gca().invert_yaxis()
    fig.savefig(os.path.join(dir,'fig2.jpg'))


    return row_ind, col_ind
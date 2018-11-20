from    copy import deepcopy
import  torch
from    torch import  nn

class Net(nn.Module):


    def __init__(self):
        super(Net, self).__init__()



        self.model = nn.Sequential(
            nn.Linear(784, 500),
            nn.ReLU(),
            nn.Linear(500, 10)
        )


    def forward(self, x):

        return self.model(x)



def weight_init(net):

    def weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight)
            torch.nn.init.constant_(m.bias, 0.0)
            # print('reseted.', m.weight.shape, m.__class__.__name__)

    for m in net.modules():
        m.apply(weights_init)


def main():

    net = Net()
    net2 = deepcopy(net)
    weight_init(net2)
    print('net:')
    for p in net.parameters():
        print(p.shape, id(p), p.norm())
    print('net2:')
    for p in net2.parameters():
        print(p.shape, id(p), p.norm())

    input = torch.randn(2, 784)
    output = net(input)
    output2 = net2(input)
    print(output.norm(), output2.norm())



if __name__ == '__main__':
    main()
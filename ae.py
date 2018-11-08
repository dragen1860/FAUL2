import  torch
import  os
import  numpy as np
from    torch import nn
from    torch.nn import functional as F
from    torch import optim
from    torch.utils.data import TensorDataset, DataLoader






class AE(nn.Module):

    def __init__(self, n_way, imgc=1, imgsz=32):
        """

        :param n_way:
        :param imgc:
        :param imgsz:
        """
        super(AE, self).__init__()


        self.n_way = n_way
        self.imgc = imgc
        self.imgsz = imgsz

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

        self.criteon = nn.MSELoss()

        h = self.encoder(torch.Tensor(2, imgc, imgsz, imgsz))
        _, self.h_c, _, self.h_d = h.size()
        out = self.decoder(h)
        print([2, imgc, imgsz, imgsz], '>:', list(h.shape), ':<', list(out.shape))

        # hidden to n_way
        self.classifier = nn.Sequential(nn.Linear(self.h_d ** 2 * self.h_c, self.n_way))


    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt: [b, sptsz, 1, 32 ,32]
        :param y_spt:
        :param x_qry: [b, qrysz, 1, 32, 32]
        :param y_qry:
        :return:
        """

        x = torch.cat([x_spt.view(-1, self.imgc, self.imgsz, self.imgsz),
                       x_qry.view(-1, self.imgc, self.imgsz, self.imgsz)], dim=0)
        # y = torch.cat([y_spt, y_qry], dim=0)

        h = self.encoder(x)
        x_hat = self.decoder(h)


        loss = self.criteon(x_hat, x)

        return loss, x_hat


    def forward_encoder(self, x):
        """

        :param x:
        :return:
        """
        return self.encoder(x)

    def forward_decoder(self, h):
        """

        :param h:
        :return:
        """
        return self.decoder(h)


    def finetuning(self, x_spt, y_spt, x_qry, y_qry, update_num):
        """
        fine-tuning on spt set and then test on query set.
        :param x_spt:
        :param y_spt:
        :param x_qry:
        :param y_qry:
        :param update_num: fine-tunning steps
        :return:
        """
        # save current state
        theta = {}
        with torch.no_grad():
            for k,v in self.state_dict().items():
                # print(k, v) # decoder.4.bias tensor([-0.0057], device='cuda:0')
                theta[k] = v.clone()


        # record original representation
        h_spt0 = self.forward_encoder(x_spt)
        h_qry0 = self.forward_encoder(x_qry)


        optimizer = optim.SGD(self.parameters(), lr=1e-1)
        losses = []
        for step in range(update_num):

            loss, x_hat = self.forward(x_spt, y_spt, x_qry, y_qry)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # now testing
        h_spt1 = self.forward_encoder(x_spt)
        h_qry1 = self.forward_encoder(x_qry)


        print('ft loss:', np.array(losses).astype(np.float16))

        # restore original state
        # make sure theta is different from updated theta
        # for k,v1 in self.state_dict().items():
        #     v0 = theta[k]
        #     print(id(v0), id(v1))
        #     print(v0.norm(p=1), v1.norm(p=1))
        self.load_state_dict(theta)


        return h_spt0, h_spt1, h_qry0, h_qry1


    def classify_reset(self):
        def weights_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.001)
                # print('reseted.', m.weight.shape, m.__class__.__name__)

        for m in self.classifier.modules():
            m.apply(weights_init)

    def classify_train(self, x_train, y_train, x_test, y_test, use_h=True, batchsz=32, train_step=50):
        """

        :param x_train:
        :param y_train:
        :param x_test:
        :param y_test:
        :param use_h:
        :param batchsz:
        :param train_step:
        :return:
        """
        # TODO: init classifier firstly
        self.classify_reset()

        criteon = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.classifier.parameters(), lr=1e-4)

        if not use_h: # given image
            # stop gradient on hidden layer
            # [b, h_c, h_d, h_d]
            x_train = self.forward_encoder(x_train).detach()
            x_test = self.forward_encoder(x_test).detach()
        else:
            x_train, x_test = x_train.detach(), x_test.detach()

        y_train, y_test = y_train.detach(), y_test.detach()

        # merge all and re-splitting
        x = torch.cat([x_train, x_test], dim=0)
        y = torch.cat([y_train, y_test], dim=0)
        db = TensorDataset(x, y)
        train_size = int(0.6 * x.size(0))
        test_size = x.size(0) - train_size
        db_train_, db_test_ = torch.utils.data.random_split(db, [train_size, test_size])

        db_train = DataLoader(db_train_, batch_size=batchsz, shuffle=True)
        db_test = DataLoader(db_test_, batch_size=batchsz, shuffle=True)

        accs = np.zeros(train_step)
        for epoch in range(train_step):
            for x, y in db_train:
                # flatten
                x = x.view(x.size(0), -1)
                logits = self.classifier(x)
                loss = criteon(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            correct, total = 0, 0
            for x, y in db_test:
                # flatten
                x = x.view(x.size(0), -1)
                logits = self.classifier(x)
                pred = logits.argmax(dim=1)
                correct += torch.eq(pred, y).sum().item()
                total += x.size(0)

            acc = correct / total
            accs[epoch] = acc

        return accs










def main():
    pass














if __name__ == '__main__':
    main()
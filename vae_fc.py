import  torch
import  os
import  numpy as np
from    torch import nn
from    torch.nn import functional as F
from    torch import optim
from    torch.utils.data import TensorDataset, DataLoader






class VAE_FC(nn.Module):

    def __init__(self, n_way, beta, q_h_d, imgc=1, imgsz=28):
        """

        :param n_way:
        :param beta: beta for vae
        :param q_h_d: different from h_c/h_d, we will convert h:[h_c, h_d, h_d] to q_h:[q_h_d]
        :param imgc:
        :param imgsz:
        """
        super(VAE_FC, self).__init__()


        self.n_way = n_way
        self.beta = beta
        self.q_h_d = q_h_d
        self.imgc = imgc
        self.imgsz = imgsz

        img_dim = imgc * imgsz * imgsz
        n_hidden = 500
        keep_prob = 1.

        # [b, imgc*imgsz*imgsz] => [b, q_h_d*2]
        self.encoder = nn.Sequential(
            nn.Linear(img_dim, n_hidden),
            nn.LeakyReLU(),
            # nn.Dropout(1-keep_prob),

            nn.Linear(n_hidden, n_hidden),
            nn.LeakyReLU(),
            # nn.Dropout(1-keep_prob),

            nn.Linear(n_hidden, q_h_d*2)

        )

        # [b, q_h_d, 1, 1] => [b, imgc, imgsz, imgsz]
        self.decoder = nn.Sequential(
            nn.Linear(q_h_d, n_hidden),
            nn.ReLU(),
            # nn.Dropout(1-keep_prob),

            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            # nn.Dropout(1-keep_prob),

            nn.Linear(n_hidden, img_dim),
            nn.Sigmoid()
        )


        # for reconstruction loss
        # self.criteon = nn.MSELoss()
        # for [0~1]
        # !!! sum is critical!
        # sum is on element-wise but NOT on img level, we need to devide by pixel number
        self.criteon = nn.BCELoss(reduction='sum')


        # hidden to n_way, based on q_h
        self.classifier = nn.Sequential(nn.Linear(self.q_h_d, self.n_way))

        print([2, imgc, imgsz, imgsz], '>:', [2, q_h_d+q_h_d], [2, q_h_d], ':<', [2, imgc, imgsz, imgsz])


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

        batchsz = x.size(0)

        # flatten
        x = x.view(batchsz, -1)

        # splitting along dim=1
        q_mu, q_sigma = self.encoder(x).chunk(2, dim=1)


        # reparametrize trick
        q_h = q_mu + q_sigma * torch.randn_like(q_sigma)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld = 0.5 * torch.sum(
                                    torch.pow(q_mu, 2) +
                                    torch.pow(q_sigma, 2) -
                                    torch.log(1e-8 + torch.pow(q_sigma, 2)) - 1
                                ).sum() / batchsz


        # with/without logits.
        x_hat = self.decoder(q_h)

        try:
            assert torch.max(x_hat) <= 1
        except:
            print(torch.max(x_hat))
            raise RuntimeError
        # reduction=sum loss to img-wise loss
        likelihood = - self.criteon(x_hat, x) / batchsz


        # elbo
        elbo = likelihood - self.beta * kld


        # restore
        x_hat = x_hat.view(batchsz, self.imgc, self.imgsz, self.imgsz)

        return -elbo, x_hat, likelihood, kld


    def forward_encoder(self, x):
        """

        :param x:
        :return:
        """
        batchsz = x.size(0)

        # flatten
        x = x.view(batchsz, -1)

        # in VAE, h here is a tmp variable
        # h actual means following q_h.
        q_mu, q_sigma = self.encoder(x).chunk(2, dim=1)

        # reparametrize trick
        q_h = q_mu + q_sigma * torch.randn_like(q_sigma)

        return q_h

    def forward_decoder(self, q_h):
        """

        :param h: it means q_h indeed
        :return:
        """
        x_hat = self.decoder(q_h)

        # x_hat = F.sigmoid(x_hat)

        # restore
        x_hat = x_hat.view(x_hat.size(0), self.imgc, self.imgsz, self.imgsz)

        return x_hat

    def finetuning(self, x_spt, y_spt, x_qry, y_qry, update_num, q_h_manifold):
        """
        fine-tuning on spt set and then test on query set.
        :param x_spt:
        :param y_spt:
        :param x_qry:
        :param y_qry:
        :param update_num: fine-tunning steps
        :param q_h_manifold: to display manifold of x
        :return:
        """
        # save current state
        theta = {}
        with torch.no_grad():
            for k,v in self.state_dict().items():
                # print(k, v) # decoder.4.bias tensor([-0.0057], device='cuda:0')
                theta[k] = v.clone()


        # record original representation
        # sampled from Normal distribution internally
        h_spt0 = self.forward_encoder(x_spt)
        h_qry0 = self.forward_encoder(x_qry)


        optimizer = optim.SGD(self.parameters(), lr=1e-3)
        losses = []
        for step in range(update_num):

            loss, x_hat, _, _ = self.forward(x_spt, y_spt, x_qry, y_qry)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # now testing
        h_spt1 = self.forward_encoder(x_spt)
        h_qry1 = self.forward_encoder(x_qry)
        x_manifold = self.forward_decoder(q_h_manifold)


        print('FT loss:', np.array(losses).astype(np.float16))

        # restore original state
        # make sure theta is different from updated theta
        # for k,v1 in self.state_dict().items():
        #     v0 = theta[k]
        #     print(id(v0), id(v1))
        #     print(v0.norm(p=1), v1.norm(p=1))
        self.load_state_dict(theta)



        return h_spt0, h_spt1, h_qry0, h_qry1, x_manifold


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
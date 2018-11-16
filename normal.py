import  torch
import  os
import  numpy as np
from    torch import nn
from    torch.nn import functional as F
from    torch import optim
from    torch.utils.data import TensorDataset, DataLoader




class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class Reshape(nn.Module):

    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(-1, *self.shape)

class AE(nn.Module):
    """
    Normal verison (Not meta) ae or vae, support fc and conv
    """

    def __init__(self, args):
        """

        :param args: 
        """
        super(AE, self).__init__()

        self.lr = args.meta_lr
        self.finetuning_lr = args.finetuning_lr
        self.classify_lr = args.classify_lr
        self.n_way = args.n_way
        self.beta = args.beta
        self.h_dim = args.h_dim
        self.imgc = args.imgc
        self.imgsz = args.imgsz
        self.is_vae = args.is_vae
        self.use_conv = args.use_conv

        img_dim = self.imgc * self.imgsz * self.imgsz

        n_hidden = 500
        keep_prob = 1.

        if self.use_conv:
            raise  NotImplementedError
        else: # fc
            if self.is_vae:
                # [b, imgc*imgsz*imgsz] => [b, q_h_d*2]
                self.encoder = nn.Sequential(
                    Flatten(),
                    nn.Linear(img_dim, n_hidden),
                    nn.LeakyReLU(),
                    # nn.Dropout(1-keep_prob),

                    nn.Linear(n_hidden, n_hidden),
                    nn.LeakyReLU(),
                    # nn.Dropout(1-keep_prob),

                    nn.Linear(n_hidden, self.h_dim*2)

                )
            else:
                # [b, imgc*imgsz*imgsz] => [b, q_h_d*2]
                self.encoder = nn.Sequential(
                    Flatten(),
                    nn.Linear(img_dim, n_hidden),
                    nn.LeakyReLU(),
                    # nn.Dropout(1-keep_prob),

                    nn.Linear(n_hidden, n_hidden),
                    nn.LeakyReLU(),
                    # nn.Dropout(1-keep_prob),

                    nn.Linear(n_hidden, self.h_dim)

                )


            # [b, q_h_d, 1, 1] => [b, imgc, imgsz, imgsz]
            self.decoder = nn.Sequential(
                nn.Linear(self.h_dim, n_hidden),
                nn.ReLU(),
                # nn.Dropout(1-keep_prob),

                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                # nn.Dropout(1-keep_prob),

                nn.Linear(n_hidden, img_dim),
                nn.Sigmoid(),

                Reshape(self.imgc, self.imgsz, self.imgsz)
            )


        # reconstruct loss
        self.criteon = nn.BCELoss(reduction='sum')
        # self.optimizer = optim.Adam(list(self.encoder.parameters())+list(self.decoder.parameters()),
        #                             lr=self.lr)

        # hidden to n_way, based on h
        self.classifier = nn.Sequential(nn.Linear(self.h_dim, self.n_way))


        print([2, self.imgc, self.imgsz, self.imgsz], '>:', [2, self.h_dim])



    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """
        forward is for training. To get ae output, you need self.forward_decoder(self.forward_encoder(x))
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


        if self.is_vae:
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

            # reduction=sum loss to img-wise loss
            likelihood = - self.criteon(x_hat, x) / batchsz
            # elbo
            elbo = likelihood - self.beta * kld

            loss = -elbo

        else:
            q_h = self.encoder(x)

            # with/without logits.
            x_hat = self.decoder(q_h)

            likelihood = kld = None

            loss = self.criteon(x_hat, x) / batchsz



        return loss, loss, likelihood, kld


    def forward_encoder(self, x):
        """

        :param x:
        :return:
        """
        batchsz = x.size(0)

        if self.is_vae:
            # in VAE, h here is a tmp variable
            # h actual means following q_h.
            q_mu, q_sigma = self.encoder(x).chunk(2, dim=1)

            # reparametrize trick
            q_h = q_mu + q_sigma * torch.randn_like(q_sigma)
        else:
            q_h = self.encoder(x)

        return q_h

    def forward_decoder(self, h):
        """

        :param h: it means q_h indeed
        :return:
        """
        x_hat = self.decoder(h)


        return x_hat

    def finetuning(self, x_spt, y_spt, x_qry, y_qry, update_num, h_manifold):
        """
        fine-tuning on spt set and then test on query set.
        :param x_spt:
        :param y_spt:
        :param x_qry:
        :param y_qry:
        :param update_num: fine-tunning steps
        :param h_manifold: to display manifold of x
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


        optimizer = optim.SGD(list(self.encoder.parameters())+list(self.decoder.parameters()),
                              lr=self.finetuning_lr)
        losses = []
        for step in range(update_num):

            loss, _, _, _ = self.forward(x_spt, y_spt, x_qry, y_qry)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # now testing
        h_spt1 = self.forward_encoder(x_spt)
        h_qry1 = self.forward_encoder(x_qry)
        x_manifold = self.forward_decoder(h_manifold)


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
                torch.nn.init.kaiming_normal_(m.weight)
                torch.nn.init.constant_(m.bias, 0.0)
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
        optimizer = optim.SGD(self.classifier.parameters(), lr=self.classify_lr)

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
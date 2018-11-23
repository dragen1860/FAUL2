import  torch
import  os
import  numpy as np
from    torch import nn
from    torch.nn import functional as F
from    torch import optim
from    torch.utils.data import TensorDataset, DataLoader
from    copy import deepcopy



class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        # print('before flaten:', x.shape)
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

    def __init__(self, args, use_logits):
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

        n_hidden = args.fc_hidden
        keep_prob = 1.

        self.use_logits = use_logits
        if use_logits:
            print('Use logits on last layer, make sure no implicit sigmoid in network config!')

        if self.use_conv:

            if self.is_vae:
                raise NotImplementedError
            else:
                self.encoder = nn.Sequential(
                    nn.Conv2d(1, args.conv_ch, kernel_size=5, stride=5, padding=0),
                    nn.BatchNorm2d(args.conv_ch),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(args.conv_ch, args.conv_ch, kernel_size=3, stride=3, padding=0),
                    nn.BatchNorm2d(args.conv_ch),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    Flatten(), # [b, 32, 1, 1]

                )

            self.decoder = nn.Sequential(
                Reshape(args.conv_ch, 1, 1),
                nn.ConvTranspose2d(args.conv_ch, args.conv_ch, kernel_size=3, stride=1, padding=0),
                nn.BatchNorm2d(args.conv_ch),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(args.conv_ch, args.conv_ch, kernel_size=3, stride=2, padding=0),
                nn.BatchNorm2d(args.conv_ch),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(args.conv_ch, args.conv_ch, kernel_size=3, stride=3, padding=0),
                nn.BatchNorm2d(args.conv_ch),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(args.conv_ch, 1, kernel_size=4, stride=3, padding=0),

            )


        else: # fc
            if self.is_vae:
                # [b, imgc*imgsz*imgsz] => [b, q_h_d*2]
                self.encoder = nn.Sequential(
                    Flatten(),
                    nn.Linear(img_dim, n_hidden),
                    nn.LeakyReLU(),
                    # nn.Dropout(1-keep_prob),

                    # nn.Linear(n_hidden//2, n_hidden//4),
                    # nn.LeakyReLU(),
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

                    # nn.Linear(n_hidden//2, n_hidden//4),
                    # nn.LeakyReLU(),
                    # nn.Dropout(1-keep_prob),

                    nn.Linear(n_hidden, self.h_dim)

                )


            # [b, q_h_d, 1, 1] => [b, imgc, imgsz, imgsz]
            self.decoder = nn.Sequential(
                nn.Linear(self.h_dim, n_hidden),
                nn.ReLU(),
                # nn.Dropout(1-keep_prob),

                # nn.Linear(n_hidden//4, n_hidden//2),
                # nn.ReLU(),
                # nn.Dropout(1-keep_prob),

                nn.Linear(n_hidden, img_dim),

                Reshape(self.imgc, self.imgsz, self.imgsz),

                # nn.Sigmoid()
            )


        # reconstruct loss
        if use_logits:
            self.criteon = nn.BCEWithLogitsLoss(reduction='elementwise_mean')
        else:
            self.criteon = nn.BCELoss(reduction='elementwise_mean')



        tmp = self.encoder(torch.Tensor(2, self.imgc, self.imgsz, self.imgsz))
        out = self.decoder(tmp)
        print('x:', [2, self.imgc, self.imgsz, self.imgsz], 'h:', tmp.shape, 'out:', out.shape, 'h_dim:', self.h_dim)
        self.h_dim = args.h_dim = tmp.size(1)
        print('overwrite h_dim from actual computation of network.')

        # hidden to n_way, based on h
        self.classifier = nn.Sequential(nn.Linear(self.h_dim, self.n_way))

        self.classify_reset(self.encoder)
        self.classify_reset(self.decoder)
        self.classify_reset(self.classifier)


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
                                    ) / np.prod(x.shape)


            # with/without logits.
            x_hat = self.decoder(q_h)

            # reduction=sum loss to img-wise loss
            likelihood = - self.criteon(x_hat, x)
            # notice: this is processed kld
            kld = self.beta * kld
            # elbo
            elbo = likelihood - kld

            loss = -elbo

        else:
            q_h = self.encoder(x)

            # with/without logits.
            x_hat = self.decoder(q_h)

            likelihood = kld = None

            loss = self.criteon(x_hat, x)

        if self.use_logits:
            # since here will not return with x_hat
            pass

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

        if self.use_logits:
            x_hat = torch.sigmoid(x_hat)

        return x_hat

    def forward_ae(self, x):
        """

        :param x:
        :return:
        """
        h = self.forward_encoder(x)
        x_hat = self.forward_decoder(h)

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
        with torch.no_grad():
            h_spt0 = self.forward_encoder(x_spt)
            h_qry0 = self.forward_encoder(x_qry)


        optimizer = optim.SGD(list(self.encoder.parameters())+list(self.decoder.parameters()),
                              lr=self.finetuning_lr)
        losses = []
        for step in range(update_num):

            loss, _, _, _ = self.forward(x_spt, y_spt, x_qry, y_qry)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(self.encoder.parameters())+list(self.decoder.parameters()), 10)
            optimizer.step()

            losses.append(loss.item())

        # now testing
        with torch.no_grad():
            h_spt1 = self.forward_encoder(x_spt)
            h_qry1 = self.forward_encoder(x_qry)

        if h_manifold is not None:
            x_manifold = self.forward_decoder(h_manifold)
        else:
            x_manifold = None



        print('FT loss:', np.array(losses).astype(np.float16))



        # create a new network model
        new_model = deepcopy(self)



        # restore original state
        # make sure theta is different from updated theta
        # for k,v1 in self.state_dict().items():
        #     v0 = theta[k]
        #     print(id(v0), id(v1))
        #     print(v0.norm(p=1), v1.norm(p=1))
        self.load_state_dict(theta)



        return h_spt0, h_spt1, h_qry0, h_qry1, x_manifold, new_model


    def classify_reset(self, net):

        def weights_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                torch.nn.init.constant_(m.bias, 0.0)
                # print('reseted.', m.weight.shape, m.__class__.__name__)

        for m in net.modules():
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
        self.classify_reset(self.classifier)

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
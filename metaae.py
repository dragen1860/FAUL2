import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np

class Learner(nn.Module):

    def __init__(self, imgsz=32, imgc = 1):
        super(Learner, self).__init__()

        # self.encoder = nn.Sequential(
        #     nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
        #     nn.ReLU(True),
        #     nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
        #     nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
        #     nn.ReLU(True),
        #     nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        # )
        #
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
        #     nn.Tanh()
        # )

        if True:
            self.config = [
                # print('ddd'*20)
                # tmp = nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1)
                # output_padding = tmp._output_padding(input, None)
                # # torch.Size([16, 8, 5, 5]) torch.Size([8]) (3, 3) (1, 1)
                # print(tmp.weight.shape, tmp.bias.shape, tmp.stride, tmp.padding,
                #       output_padding, tmp.groups, tmp.dilation) # (0, 0) 1 (1, 1)
                # # stride=1, padding=0, output_padding=0, groups=1, dilation=1
                # F.conv_transpose2d()
                # print('ddd'*20)

                # conv2d:[c_out, c_in, kernelsz, kernelsz, stride, padding]
                # convt2d: [c_in, c_out, kernelsz, kernelsz, stride, padding], output_padding=0, groups=1, dilation=1
                # pool2d:[kernelsz, stride, padding]
                ('conv2d', [16, 1, 3, 3, 3, 1]), # the first
                ('relu',   [True]),
                ('max_pool2d', [2, 2, 0]),
                ('conv2d', [8, 16, 3, 3, 2, 1]), #
                ('relu',[True]),
                ('max_pool2d', [2, 1, 0]),


                ('hidden', []), # hidden variable

                # [ch_out, ch_in]
                ('convt2d', [8, 16, 3, 3, 2, 0]),    # defactor1
                ('relu', [True]),
                ('convt2d', [16, 8, 5, 5, 3, 1]),
                ('relu', [True]),
                ('convt2d', [8, 1, 2, 2, 2, 1]),
                ('tanh',[])
            ]
        else:
            self.config = [
                # conv2d:[c_out, c_in, kernelsz, kernelsz, stride, padding]
                # pool2d:[kernelsz, stride, padding]
                # upsample:[factor]
                ('conv2d', [16, 1, 1, 1, 1, 1]),  # the first

                ('conv2d', [16, 16, 3, 3, 1, 0]),  # factor1
                ('avg_pool2d', [2, 2, 0]),  # factor1
                ('conv2d', [16, 16, 3, 3, 1, 1]),  # factor1
                ('avg_pool2d', [2, 2, 0]),  # factor1

                ('conv2d', [4, 16, 3, 3, 1, 1]),  #

                ('hidden', []),  # hidden variable
                ('conv2d', [16, 4, 3, 3, 1, 1]),  # defactor1
                ('upsample', [2]),
                ('conv2d', [16, 16, 3, 3, 1, 1]),  # defactor1
                ('upsample', [2]),

                # ('conv2d', [16, 16, 3, 3, 1, 1]),
                ('conv2d', [1, 16, 3, 3, 1, 1])
            ]

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()

        for name, param in self.config:
            if name is 'conv2d':
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.xavier_uniform_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name is 'convt2d':
                # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.xavier_uniform_(w)
                self.vars.append(w)
                # [ch_in, ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[1])))

            elif name is 'tanh':
                continue
            elif name is 'relu':
                continue
            elif name is 'hidden':
                continue
            elif name is 'upsample':
                continue
            elif name is 'avg_pool2d':
                continue
            elif name is 'max_pool2d':
                continue
            else:
                raise NotImplementedError

        h = self.forward_encoder(torch.Tensor(2, imgc, imgsz, imgsz))
        out = self.forward(torch.Tensor(2, imgc, imgsz, imgsz))
        print('hidden:', h.shape, 'out:', out.shape)
        _, self.h_c, self.h_d, _ = h.shape


    def extra_repr(self):
        info = ''

        for name, param in self.config:
            if name is 'conv2d':
                tmp = 'conv2d:(ch_out:%d, ch_in:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[0], param[1], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'convt2d':
                tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[0], param[1], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'tanh':
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            elif name is 'relu':
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            elif name is 'hidden':
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            elif name is 'upsample':
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            elif name is 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name is 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            else:
                raise NotImplementedError

        return info



    def forward(self, x, vars=None):

        if vars is None:
            vars = self.vars

        idx = 0

        for name, param in self.config:
            if name is 'conv2d':
                w, b = vars[idx:(idx + 2)]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name is 'convt2d':
                w, b = vars[idx:(idx + 2)]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name is 'relu':
                x = F.relu(x, inplace=param[0])
            elif name is 'tanh':
                x = F.tanh(x)
            elif name is 'hidden':
                continue
            elif name is 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])
            elif name is 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])
            elif name is 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])
            else:
                raise NotImplementedError

        # make sure variable is used properly
        assert idx == len(self.vars)

        return x

    def forward_encoder(self, x, vars=None):
        """
        forward till hidden layer
        :param x:
        :param vars:
        :return:
        """
        if vars is None:
            vars = self.vars

        idx = 0

        for name, param in self.config:
            if name is 'conv2d':
                w, b = vars[idx:(idx + 2)]
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
            elif name is 'convt2d':
                w, b = vars[idx:(idx + 2)]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name is 'relu':
                x = F.relu(x, inplace=param[0])
            elif name is 'tanh':
                x = F.tanh(x)
            elif name is 'hidden':
                break
            elif name is 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])

            elif name is 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])
            elif name is 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])
            else:
                raise NotImplementedError


        return x

    def forward_decoder(self, h, vars=None):
        """
        forward after hidden layer
        :param x:
        :param vars:
        :return:
        """
        if vars is None:
            vars = self.vars


        hidden_loc = 0
        for name, param in self.config:
             if name is not 'hidden':
                 hidden_loc += 1

        if hidden_loc == len(self.config):
            raise NotImplementedError

        # get decoder network config
        decoder_config = self.config[hidden_loc+1:]


        idx = 0
        x = h
        for name, param in decoder_config:
            if name is 'conv2d':
                w, b = vars[idx:(idx + 2)]
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
            elif name is 'convt2d':
                w, b = vars[idx:(idx + 2)]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name is 'tanh':
                x = F.tanh(x)
            elif name is 'relu':
                x = F.relu(x, inplace=param[0])
            elif name is 'hidden':
                raise NotImplementedError
            elif name is 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])
            elif name is 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])
            elif name is 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])
            else:
                raise NotImplementedError


        return x

    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars


class MetaLearner(nn.Module):

    def __init__(self, n_way, k_spt, k_qry, task_num, update_num, meta_lr, update_lr):
        """

        :param n_way:
        :param k_spt:
        :param k_qry:
        :param task_num:
        :param update_num:
        :param meta_lr:
        :param update_lr:
        """
        super(MetaLearner, self).__init__()

        self.update_lr = update_lr
        self.meta_lr = meta_lr
        self.n_way = n_way
        self.k_spt = k_spt
        self.k_qry = k_qry
        self.task_num = task_num
        self.update_num = update_num

        self.learner = Learner()
        self.criteon = nn.MSELoss()
        self.meta_optim = optim.Adam(self.learner.parameters(), lr=self.meta_lr)

        # hidden to n_way
        self.classifier = nn.Sequential(nn.Linear(self.learner.h_d**2 * self.learner.h_c, self.n_way))

        print(self.learner)

    def classify_reset(self):
        def weights_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.001)
                # print('reseted.', m.weight.shape, m.__class__.__name__)

        for m in self.classifier.modules():
            m.apply(weights_init)


    def finetuning(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt: [task_num, sptsz, c_, h, w]
        :param y_spt: [task_num, sptsz]
        :param x_qry:
        :param y_qry:
        :return:
        """
        sptsz, c_, h, w = x_spt.size()
        qrysz = x_qry.size(0)
        assert len(x_spt.shape) == 4
        assert len(x_qry.shape) == 4

        # use theta to forward
        pred = self.learner(x_spt)
        loss = self.criteon(pred, x_spt)

        # 2. grad on theta
        # clear theta grad info
        self.learner.zero_grad()
        grad = torch.autograd.grad(loss, self.learner.parameters())

        # 3. theta_pi = theta - train_lr * grad
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.learner.parameters())))


        # 4. continue to update
        for k in range(1, self.update_num):
            # 1. run the i-th task and compute loss for k=1~K-1
            pred = self.learner(x_spt, fast_weights)
            loss = self.criteon(pred, x_spt)
            # clear fast_weights grad info
            self.learner.zero_grad(fast_weights)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))


        # TODO:
        with torch.no_grad():
            # 5. acquire representation
            h_spt0 = self.learner.forward_encoder(x_spt)
            h_spt1 = self.learner.forward_encoder(x_spt, fast_weights)
            h_qry0 = self.learner.forward_encoder(x_qry)
            h_qry1 = self.learner.forward_encoder(x_qry, fast_weights)

        return h_spt0, h_spt1, h_qry0, h_qry1

    def classify_train(self, x_train, y_train, x_test, y_test, use_h=True, batchsz=32, train_step=50):
        """

        :param x_train: [b, c_, h, w]
        :param y_train: [b]
        :param x_test:
        :param y_test:
        :param use_h: use h or x
        :param batchsz: batchsz for classifier
        :param train_step: training steps for classifier
        :return:
        """
        # TODO: init classifier firstly
        self.classify_reset()

        criteon = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.classifier.parameters(), lr=1e-4)

        if not use_h: # given image
            # stop gradient on hidden layer
            # [b, h_c, h_d, h_d]
            x_train = self.learner.forward_encoder(x_train).detach()
            x_test = self.learner.forward_encoder(x_test).detach()
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





    def forward(self, x_spt, y_spt, x_qry, y_qry, training=True):
        """

        :param x_spt: [task_num, setsz, c_, h, w]
        :param y_spt: [task_num, setsz]
        :param x_qry:
        :param y_qry:
        :param training:
        :return:
        """
        # batchsz = task_num
        batchsz, sptsz, c_, h, w = x_spt.size()
        qrysz = x_qry.size(1)

        losses_q = []  # losses_q[i], i is tasks idx

        # TODO: add multi-threading support
        # NOTICE: although the GIL limit the multi-threading performance severely, it does make a difference.
        # When deal with IO operation,
        # we need to coordinate with outside IO devices. With the assistance of multi-threading, we can issue multi-commands
        # parallelly and improve the efficency of IO usage.
        for i in range(batchsz):

            # 1. run the i-th task and compute loss for k=0
            pred = self.learner(x_spt[i])
            loss = self.criteon(pred, x_spt[i])

            # 2. grad on theta
            # clear theta grad info
            self.learner.zero_grad()
            grad = torch.autograd.grad(loss, self.learner.parameters())

            # print('k0')
            # for p in grad[:5]:
            # 	print(p.norm().item())

            # 3. theta_pi = theta - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.learner.parameters())))

            # this is the loss and accuracy before first update
            # [setsz, nway]
            pred_q = self.learner(x_qry[i], self.learner.parameters())

            # this is the loss and accuracy after the first update
            # [setsz, nway]
            pred_q = self.learner(x_qry[i], fast_weights)
            loss_q = self.criteon(pred_q, x_qry[i])

            for k in range(1, self.update_num):
                # 1. run the i-th task and compute loss for k=1~K-1
                pred = self.learner(x_spt[i], fast_weights)
                loss = self.criteon(pred, x_spt[i])
                # clear fast_weights grad info
                self.learner.zero_grad(fast_weights)
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                pred_q = self.learner(x_qry[i], fast_weights)
                loss_q = self.criteon(pred_q, x_qry[i])

            # 4. record last step's loss for task i
            losses_q.append(loss_q)

        # end of all tasks
        # sum over all losses across all tasks
        loss_q = torch.stack(losses_q).sum(0)

        self.meta_optim.zero_grad()
        loss_q.backward()
        # print('meta update')
        # for p in self.learner.parameters()[:5]:
        # 	print(torch.norm(p).item())
        self.meta_optim.step()

        return loss_q



def main():
    net = MetaLearner(5, 1, 15, 8, 5, 1e-3, 0.05)
    print(net)


if __name__ == '__main__':
    main()

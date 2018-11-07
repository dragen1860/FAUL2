import torch
from torch import nn
from torch import optim
from torch.nn import functional as F


class Learner(nn.Module):

    def __init__(self):
        super(Learner, self).__init__()

        self.config = [
            ('conv2d', [16, 1, 1, 1]),
            ('conv2d', [16, 16, 3, 3]),
            ('conv2d', [16, 16, 3, 3]),
            ('conv2d', [8, 16, 3, 3]),

            ('conv2d', [16, 8, 3, 3]),
            ('conv2d', [16, 16, 3, 3]),
            ('conv2d', [16, 16, 3, 3]),
            ('conv2d', [1, 16, 3, 3])
        ]

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()

        for name, param in self.config:
            if name is 'conv2d':
                # [ch_out, ch_in, kernelsz, kernelsz]
                self.vars.append(nn.Parameter(torch.ones(*param)))
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name is 'linear':
                raise NotImplementedError
            else:
                raise NotImplementedError

    def extra_repr(self):
        info = ''

        for name, param in self.config:
            if name is 'conv2d':
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            elif name is 'linear':
                raise NotImplementedError
            else:
                raise NotImplementedError

        return info

    def forward(self, x, vars):

        if vars is None:
            vars = self.vars

        idx = 0

        for name, param in self.config:
            if name is 'conv2d':
                w, b = vars[idx:(idx + 2)]
                x = F.conv2d(x, w, b, stride=1, padding=0)
                idx += 2


            elif name is 'linear':
                raise NotImplementedError
            else:
                raise NotImplementedError

        # make sure variable is used properly
        assert idx == len(self.vars)

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

    def forward(self, x_spt, y_spt, x_qry, y_qry, training):
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
        corrects = [0] * (self.K + 1)  # corrects[i] save cumulative correct number of all tasks in step k

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
            pred_q = self.learner(x_qry[i], self.net.parameters())

            # this is the loss and accuracy after the first update
            # [setsz, nway]
            pred_q = self.learner(x_qry[i], fast_weights)
            loss_q = self.criteon(pred_q, x_qry[i])

            for k in range(1, self.update_num):
                # 1. run the i-th task and compute loss for k=1~K-1
                pred = self.net(x_spt[i], fast_weights)
                loss = self.criteon(pred, x_spt[i])
                # clear fast_weights grad info
                self.net.zero_grad(fast_weights)
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                pred_q = self.net(x_qry[i], fast_weights)
                loss_q = self.criteon(pred_q, x_qry[i])

            # 4. record last step's loss for task i
            losses_q.append(loss_q)

        # end of all tasks
        # sum over all losses across all tasks
        loss_q = torch.stack(losses_q).sum(0)

        self.meta_optim.zero_grad()
        loss_q.backward()
        # print('meta update')
        # for p in self.net.parameters()[:5]:
        # 	print(torch.norm(p).item())
        self.meta_optim.step()


def main():
    net = MetaLearner(5, 1, 15, 8, 5, 1e-3, 0.05)
    print(net)


if __name__ == '__main__':
    main()

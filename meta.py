import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np

from    learner import AELearner


class MetaAE(nn.Module):
    """
    Meta version of vae or ae, supporting fc and conv.
    """
    def __init__(self, args):
        """

        :param args:
        """
        super(MetaAE, self).__init__()

        self.update_lr = args.update_lr
        self.classify_lr = args.classify_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_num = args.update_num
        self.img_dim = args.imgc * args.imgsz * args.imgsz
        self.is_vae = args.is_vae
        self.use_conv = args.use_conv


        if self.use_conv:
            raise NotImplementedError
        else:
            if self.is_vae:
                config = [
                    ('flatten', []),
                    ('linear', [500, self.img_dim]),
                    ('leakyrelu', [0.01, True]),
                    ('linear', [500, 500]),
                    ('leakyrelu', [0.01, True]),
                    ('linear', [2* args.h_dim, 500]),

                    ('hidden', []),

                    ('linear', [500, args.h_dim]),
                    ('relu', [True]),
                    ('linear', [500, 500]),
                    ('relu', [True]),
                    ('linear', [self.img_dim, 500]),
                    ('reshape', [args.imgc, args.imgsz, args.imgsz]),
                    ('use_logits',[])

                ]
            else:
                config = [
                    ('flatten', []),
                    ('linear', [500, self.img_dim]),
                    ('leakyrelu', [0.01, True]),
                    ('linear', [500, 500]),
                    ('leakyrelu', [0.01, True]),
                    ('linear', [args.h_dim, 500]),

                    ('hidden', []),

                    ('linear', [500, args.h_dim]),
                    ('relu', [True]),
                    ('linear', [500, 500]),
                    ('relu', [True]),
                    ('linear', [self.img_dim, 500]),
                    ('reshape', [args.imgc, args.imgsz, args.imgsz]),
                    ('use_logits',[]),

                ]

        self.learner = AELearner(config, args.imgc, args.imgsz, is_vae=args.is_vae, beta=args.beta)
        self.meta_optim = optim.Adam(self.learner.parameters(), lr=self.meta_lr)

        # hidden to n_way
        self.classifier = nn.Sequential(nn.Linear(args.h_dim, self.n_way))



    def classify_reset(self):
        """
        reset classifier weights before each classification.
        :return:
        """
        def weights_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                torch.nn.init.constant_(m.bias, 0.0)
                # print('reseted.', m.weight.shape, m.__class__.__name__)

        for m in self.classifier.modules():
            m.apply(weights_init)


    def finetuning(self, x_spt, y_spt, x_qry, y_qry, update_num, h_manifold=None):
        """

        :param x_spt: [task_num, sptsz, c_, h, w]
        :param y_spt: [task_num, sptsz]
        :param x_qry:
        :param y_qry:
        :param update_num: update for fine-tuning
        :param h_manifold: [b, 2] manifold of h
        :return:
        """
        sptsz, c_, h, w = x_spt.size()
        qrysz = x_qry.size(0)
        assert len(x_spt.shape) == 4
        assert len(x_qry.shape) == 4

        # use theta to forward
        pred, loss, likelihood, kld = self.learner(x_spt)

        # 2. grad on theta
        # clear theta grad info
        self.learner.zero_grad()
        grad = torch.autograd.grad(loss, self.learner.parameters())

        # 3. theta_pi = theta - train_lr * grad
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.learner.parameters())))


        # 4. continue to update
        for k in range(1, update_num):
            # 1. run the i-th task and compute loss for k=1~K-1
            pred, loss, likelihood, kld = self.learner(x_spt, fast_weights)
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

            if h_manifold is not None:
                x_manifold = self.learner.forward_decoder(h_manifold, fast_weights)
            else:
                x_manifold = None

        return h_spt0, h_spt1, h_qry0, h_qry1, x_manifold

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
        optimizer = optim.SGD(self.classifier.parameters(), lr=self.classify_lr)

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


    def forward_encoder(self, x):
        """

        :param x:
        :return:
        """
        return self.learner.forward_encoder(x)

    def forward_decoder(self, h):
        """

        :param h:
        :return:
        """
        return self.learner.forward_decoder(h)


    def forward(self, x_spt, y_spt, x_qry, y_qry):
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

        # statistic data for qry
        # NOTICE: it's different from [[]]*update_num !!!
        losses_q, likelihoods_q, klds_q = [[] for _ in range(self.update_num+1)], \
                                          [[] for _ in range(self.update_num+1)], \
                                          [[] for _ in range(self.update_num+1)]
        # statistic data for spt
        # for spt, it only has update_num items
        losses_p, likelihoods_p, klds_p = [[] for _ in range(self.update_num+1)], \
                                          [[] for _ in range(self.update_num+1)],\
                                          [[] for _ in range(self.update_num+1)]

        def update_statistic(loss, likelihood, kld, loss_q, likelihood_q, kld_q, k):
            losses_p[k].append(loss)
            likelihoods_p[k].append(likelihood)
            klds_p[k].append(kld)
            losses_q[k].append(loss_q)
            likelihoods_q[k].append(likelihood_q)
            klds_q[k].append(kld_q)




        # TODO: add multi-threading support
        # NOTICE: although the GIL limit the multi-threading performance severely, it does make a difference.
        # When deal with IO operation,
        # we need to coordinate with outside IO devices. With the assistance of multi-threading, we can issue multi-commands
        # parallelly and improve the efficency of IO usage.
        for i in range(batchsz):

            # this is the loss and accuracy before first update
            pred_q0, loss_q0, likelihood_q0, kld_q0 = self.learner(x_qry[i])
            update_statistic(None, None, None, loss_q0, likelihood_q0, kld_q0, 0)

            # 1. run the i-th task and compute loss for k=0
            pred, loss, likelihood, kld = self.learner(x_spt[i])


            # 2. grad on theta
            # clear theta grad info
            self.learner.zero_grad()
            grad = torch.autograd.grad(loss, self.learner.parameters())

            # print('k0')
            # for p in grad[:5]:
            # 	print(p.norm().item())

            # 3. theta_pi = theta - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.learner.parameters())))


            # this is the loss and accuracy after the first update
            pred_q, loss_q, likelihood_q, kld_q = self.learner(x_qry[i], fast_weights)
            update_statistic(loss, likelihood, kld, loss_q, likelihood_q, kld_q, 1)

            for k in range(1, self.update_num):
                # 1. run the i-th task and compute loss for k=1~K-1
                pred, loss, likelihood, kld = self.learner(x_spt[i], fast_weights)
                # clear fast_weights grad info
                self.learner.zero_grad(fast_weights)
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                pred_q, loss_q, likelihood_q, kld_q = self.learner(x_qry[i], fast_weights)

                update_statistic(loss, likelihood, kld, loss_q, likelihood_q, kld_q, k+1)


        if self.is_vae:
            # get mean loss across tasks on each step, ignore the loss before update on spt.
            # obj = [[step0_task1, step0_task2,....], [step2,...]]
            # 0-dim tensor can not be cated, only be stacked!
            losses_p = [torch.stack(step).mean() for step in losses_p[1:]]
            likelihoods_p = [torch.stack(step).mean() for step in likelihoods_p[1:]]
            klds_p = [torch.stack(step).mean() for step in klds_p[1:]]
            losses_q = [torch.stack(step).mean() for step in losses_q]
            likelihoods_q = [torch.stack(step).mean() for step in likelihoods_q]
            klds_q = [torch.stack(step).mean() for step in klds_q]

            # end of all tasks
            # scalar tensor.
            loss_optim = losses_q[-1]

            self.meta_optim.zero_grad()
            loss_optim.backward()
            # print('meta update')
            # for p in self.learner.parameters()[:5]:
            # 	print(torch.norm(p).item())
            self.meta_optim.step()

            return loss_optim, losses_q, likelihoods_q, klds_q

        else: # for ae, likelihood and klds= None
            losses_p = [torch.stack(step).mean() for step in losses_p[1:]]
            losses_q = [torch.stack(step).mean() for step in losses_q]

            # end of all tasks
            # scalar tensor.
            loss_optim = losses_q[-1]

            self.meta_optim.zero_grad()
            loss_optim.backward()
            print('meta update')
            for p in self.learner.parameters()[:5]:
                print(torch.norm(p.grad).item())
            self.meta_optim.step()

            return loss_optim, losses_q, None, None



def main():
    pass


if __name__ == '__main__':
    main()
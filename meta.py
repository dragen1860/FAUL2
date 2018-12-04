import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np

from    learner import AELearner
from    copy import deepcopy
from    sklearn import cluster, metrics



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

        fc_hidden = args.fc_hidden



        if self.use_conv:
            if self.is_vae:
                raise NotImplementedError
            else:
                config = [
                    ('conv2d', [args.conv_ch, 1, 5, 5, 5, 0]),  # the first
                    ('bn', [args.conv_ch]),
                    ('relu', [True]),
                    ('max_pool2d', [2, 2, 0]),
                    ('conv2d', [args.conv_ch, args.conv_ch, 3, 3, 3, 0]),
                    ('bn', [args.conv_ch]),
                    ('relu', [True]),
                    ('max_pool2d', [2, 2, 0]),
                    ('flatten', []),

                    ('hidden', []),  # hidden variable

                    # [ch_out, ch_in]
                    ('reshape', [args.conv_ch, 1, 1]),
                    ('convt2d', [args.conv_ch, args.conv_ch, 3, 3, 1, 0]),
                    ('bn', [args.conv_ch]),
                    ('relu', [True]),
                    ('convt2d', [args.conv_ch, args.conv_ch, 3, 3, 2, 0]),
                    ('bn', [args.conv_ch]),
                    ('relu', [True]),
                    ('convt2d', [args.conv_ch, args.conv_ch, 3, 3, 3, 0]),
                    ('bn', [args.conv_ch]),
                    ('relu', [True]),
                    ('convt2d', [args.conv_ch, 1, 4, 4, 3, 0]),
                    ('use_logits', [])
                ]
        else: # fully-connected
            if self.is_vae:
                config = [
                    ('flatten', []),
                    ('linear', [fc_hidden, self.img_dim]),
                    ('leakyrelu', [0.02, True]),
                    # ('linear', [fc_hidden//4, fc_hidden//2]),
                    # ('leakyrelu', [0.02, True]),
                    ('linear', [2* args.h_dim, fc_hidden]),
                    # ('usigma_layer', [args.h_dim, 500]),

                    ('hidden', []),

                    ('linear', [fc_hidden, args.h_dim]),
                    ('relu', [True]),
                    # ('linear', [fc_hidden//2, fc_hidden//4]),
                    # ('relu', [True]),
                    ('linear', [self.img_dim, fc_hidden]),
                    ('reshape', [args.imgc, args.imgsz, args.imgsz]),
                    # ('sigmoid', []),
                    ('use_logits',[]) # sigmoid with logits loss

                ]
            else:
                config = [
                    ('flatten', []),
                    ('linear', [fc_hidden, self.img_dim]),
                    ('leakyrelu', [0.02, True]),
                    # ('linear', [fc_hidden//4, fc_hidden//2]),
                    # ('leakyrelu', [0.02, True]),
                    ('linear', [args.h_dim, fc_hidden]),

                    ('hidden', []),

                    ('linear', [fc_hidden, args.h_dim]),
                    ('relu', [True]),
                    # ('linear', [fc_hidden//2, fc_hidden//4]),
                    # ('relu', [True]),
                    ('linear', [self.img_dim, fc_hidden]),
                    ('reshape', [args.imgc, args.imgsz, args.imgsz]),
                    # ('sigmoid', []),
                    ('use_logits',[]), # sigmoid with logits loss

                ]

        self.learner = AELearner(config, args.imgc, args.imgsz, is_vae=args.is_vae, beta=args.beta)
        self.meta_optim = optim.Adam(self.learner.parameters(), lr=self.meta_lr)

        # hidden to n_way
        h = self.forward_encoder(torch.randn(2, args.imgc, args.imgsz, args.imgsz))
        args.h_dim = h.size(1)
        print('overwrite h_dim from actual computation of network.')
        self.classifier = nn.Sequential(nn.Linear(args.h_dim, self.n_way))

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

    def forward_ae(self, x):
        """

        :param x:
        :return:
        """
        return self.learner.forward_ae(x)

    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter

    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt: [task_num, setsz, c_, h, w]
        :param y_spt: [task_num, setsz]
        :param x_qry:
        :param y_qry:
        :return:
        """
        # batchsz = task_num
        meta_batchsz, sptsz, c_, h, w = x_spt.size()
        qrysz = x_qry.size(1)

        # statistic data for qry
        # NOTICE: it's different from [[]]*update_num !!!
        losses_q, likelihoods_q, klds_q = [[] for _ in range(self.update_num + 1)], \
                                          [[] for _ in range(self.update_num + 1)], \
                                          [[] for _ in range(self.update_num + 1)]
        # statistic data for spt
        # for spt, it only has update_num items
        losses_p, likelihoods_p, klds_p = [[] for _ in range(self.update_num + 1)], \
                                          [[] for _ in range(self.update_num + 1)], \
                                          [[] for _ in range(self.update_num + 1)]

        def update_statistic(loss, likelihood, kld, loss_q, likelihood_q, kld_q, k):
            """
            save all intermediate statistics into list.
            :param loss:
            :param likelihood:
            :param kld:
            :param loss_q:
            :param likelihood_q:
            :param kld_q:
            :param k: step k-1
            :return:
            """
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
        for i in range(meta_batchsz):

            # this is the loss and accuracy before first update
            with torch.no_grad():
                pred_q0, loss_q0, likelihood_q0, kld_q0 = self.learner(x_qry[i])
                update_statistic(None, None, None, loss_q0, likelihood_q0, kld_q0, 0)

            # 1. run the i-th task and compute loss for k=0
            pred, loss, likelihood, kld = self.learner(x_spt[i])

            # 2. grad on theta
            # clear theta grad info
            self.learner.zero_grad()
            grad = torch.autograd.grad(loss, self.learner.parameters())
            self.clip_grad_by_norm_(grad, 10)


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
                total_norm = self.clip_grad_by_norm_(grad, 10)


                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))


                pred_q, loss_q, likelihood_q, kld_q = self.learner(x_qry[i], fast_weights)
                update_statistic(loss, likelihood, kld, loss_q, likelihood_q, kld_q, k + 1)

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

        else:  # for ae, likelihood and klds= None
            losses_p = [torch.stack(step).mean() for step in losses_p[1:]]
            losses_q = [torch.stack(step).mean() for step in losses_q]

            # end of all tasks
            # scalar tensor.
            loss_optim = losses_q[-1]

            self.meta_optim.zero_grad()
            loss_optim.backward()
            # print('meta update')
            # for p in self.learner.parameters()[:5]:
            #     print(torch.norm(p.grad).item())
            self.meta_optim.step()

            return loss_optim, losses_q, None, None



    def classify_reset(self, net):
        """
        reset classifier weights before each classification.
        :return:
        """
        def weights_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                torch.nn.init.constant_(m.bias, 0.0)
                # print('reseted.', m.weight.shape, m.__class__.__name__)

        for m in net.modules():
            m.apply(weights_init)


    def finetuning(self, x_spt, y_spt, x_qry, y_qry, update_num, h_manifold=None):
        """
        finetunning will update weights/bias of batch_norm as well, but the running_mean
        and running_var will not be updated since we set training=False?

        In order keep state of intial theta network, we can use fast_weighs to separate from updated
        weights/bias from original. But running_mean/vars will still be updated in learner.forward() function.
        Therefore, we use  update_bn_statistics=False to force no updating of running_mean/vars in finnuting phase.
        This will not affect normal training phase.
        :param x_spt: [sptsz, c_, h, w]
        :param y_spt: [sptsz]
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


        losses = []
        # use theta to forward
        pred, loss, likelihood, kld = self.learner(x_spt, update_bn_statistics=False)
        losses.append(loss.item())
        h_qry1_amis, h_qry1_arss = [], []

        # 2. grad on theta
        # clear theta grad info
        self.learner.zero_grad()
        grad = torch.autograd.grad(loss, self.learner.parameters())
        self.clip_grad_by_norm_(grad, 10)

        # 3. theta_pi = theta - train_lr * grad
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.learner.parameters())))


        # 4. continue to update
        for k in range(1, update_num):
            # 1. run the i-th task and compute loss for k=1~K-1
            pred, loss, likelihood, kld = self.learner(x_spt, fast_weights, update_bn_statistics=False)
            losses.append(loss.item())
            # clear fast_weights grad info
            self.learner.zero_grad(fast_weights)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            self.clip_grad_by_norm_(grad, 10)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            with torch.no_grad():
                # [qrysz, 1, 64, 64] => [qrysz, d_dim]
                h_qry1 = self.learner.forward_encoder(x_qry, fast_weights)
                h_qry1_np = h_qry1.detach().cpu().numpy()
                # [qrysz]
                qry_y_np = y_qry.detach().cpu().numpy()
                h_qry1_pred = cluster.KMeans(n_clusters=self.n_way, random_state=1).fit(h_qry1_np).labels_
                h_qry1_amis.append(metrics.adjusted_mutual_info_score(qry_y_np, h_qry1_pred))
                h_qry1_arss.append(metrics.adjusted_rand_score(qry_y_np, h_qry1_pred))


        print('Loss:', np.array(losses).astype(np.float16))
        print('AMI :', np.array(h_qry1_amis).astype(np.float16))
        print('ARS :', np.array(h_qry1_arss).astype(np.float16))


        # running_mean, running_var = self.learner.vars_bn[0], self.learner.vars_bn[1]
        # print('after :', running_mean.norm().item(), running_var.norm().item())


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

            # establish a new learner and copy the weights into it
            new_learner = deepcopy(self.learner)
            new_learner.vars = nn.ParameterList(map(lambda x:nn.Parameter(x), fast_weights))
            # for p, new_p in zip(new_learner.parameters(), fast_weights):
            #     # copy_ need same type, however nn.Parameter vs FloatTensor
            #     p.data = new_p.data

        return h_spt0, h_spt1, h_qry0, h_qry1, x_manifold, new_learner

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

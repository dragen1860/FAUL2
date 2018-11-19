import  argparse
import  torch
from    torch import optim
from    torch.utils.data import DataLoader
import  numpy as np
from    matplotlib import pyplot as plt
import  visdom

from    meta import MetaAE
from    normal import AE
from    mnistNShot import MnistNShot

from    visualization import VisualH


def update_args(args):

    if args.exp is None:
        # once exp is not specified, we use default config.
        return args

    exp = args.exp

    if exp == 'meta-conv-ae':
        args.is_vae = False
        args.is_meta = True
        args.use_conv = True
        args.task_num = 4
        args.meta_lr = 1e-3
        args.update_num = 5
        args.update_lr = 0.01
        args.finetuning_lr = 0.01
        args.finetuning_steps = 15
        args.classify_steps = 10
        args.classify_lr = 0.01
        args.h_dim = 8*2*2

    elif exp == 'meta-conv-vae':
        args.is_vae = True
        args.is_meta = True
        args.use_conv = True
        args.task_num = 4
        args.meta_lr = 1e-3
        args.update_num = 5
        args.update_lr = 0.01
        args.finetuning_lr = 0.01
        args.finetuning_steps = 15
        args.classify_steps = 10
        args.classify_lr = 0.01
        args.h_dim = 2

    elif exp == 'meta-fc-ae':
        args.is_vae = False
        args.is_meta = True
        args.use_conv = False
        args.task_num = 4
        args.meta_lr = 1e-3
        args.update_num = 5
        args.update_lr = 0.01
        args.finetuning_lr = 0.01
        args.finetuning_steps = 15
        args.classify_steps = 10
        args.classify_lr = 0.01
        args.h_dim = 10

    elif exp == 'meta-fc-vae':
        args.is_vae = True
        args.is_meta = True
        args.use_conv = False
        args.task_num = 4
        args.meta_lr = 1e-3
        args.update_num = 5
        args.update_lr = 0.01
        args.finetuning_lr = 0.01
        args.finetuning_steps = 15
        args.classify_steps = 10
        args.classify_lr = 0.01
        args.h_dim = 10


    elif exp == 'normal-fc-ae':
        args.is_vae = False
        args.is_meta = False
        args.use_conv = False
        args.task_num = 4
        args.meta_lr = 1e-3 # learning rate
        args.update_num = 5 # no use
        args.update_lr = 0.01 # no use
        args.finetuning_lr = 0.01
        args.finetuning_steps = 15
        args.classify_steps = 10
        args.classify_lr = 0.01
        args.h_dim = 10

    elif exp == 'normal-fc-vae':
        args.is_vae = True
        args.beta = 0.5
        args.is_meta = False
        args.use_conv = False
        args.task_num = 4
        args.meta_lr = 1e-3 # learning rate
        args.update_num = 5 # no use
        args.update_lr = 0.01 # no use
        args.finetuning_lr = 0.001
        args.finetuning_steps = 15
        args.classify_steps = 10
        args.classify_lr = 0.01
        args.h_dim = 10

    else:
        print('Wrong Exp name:', exp)
        raise NotImplementedError

    return args


def main(args):

    args = update_args(args)

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    device = torch.device('cuda')
    if args.is_meta:
        # optimizer has been embedded in model.
        net = MetaAE(args)
    else:
        net = AE(args, use_logits=True)
        optimizer = optim.Adam(list(net.encoder.parameters()) + list(net.decoder.parameters()),
                               lr=args.meta_lr)

    print(net)
    net.to(device)

    task_name = ''.join(['meta' if args.is_meta else 'normal','-',
                        'conv' if args.use_conv else 'fc','-',
                        'vae' if args.is_vae else 'ae'])
    print('='*15,'Experiment:', task_name, '='*15)
    print(args)

    vis = visdom.Visdom(env=task_name)
    visualh = VisualH(vis)
    global_step = 0
    vis.line([[0,0,0]], [0], win='train_loss', opts=dict(
                                                    title='train_loss',
                                                    legend=['loss', '-lklh', 'kld'])
             )
    vis.line([[0,0]], [[0,0]], win='classify_acc', opts=dict(legend=['before', 'after'],
                                                             showlegend=True,
                                                             title='class_acc'))

    if args.h_dim == 2:
        # borrowed from https://github.com/fastforwardlabs/vae-tf/blob/master/plot.py
        h_range = np.rollaxis(np.mgrid[args.h_range:-args.h_range:args.h_nrow * 1j,
                                args.h_range:-args.h_range:args.h_nrow * 1j], 0, 3)
        # [b, q_h]
        h_manifold = torch.from_numpy(h_range.reshape([-1, 2])).to(device).float()
        print('h_manifold:', h_manifold.shape)
    else:
        h_manifold = None


    for epoch in range(args.epoch):

        # 1. train
        db_train = DataLoader(
            MnistNShot('db/mnist', training=True, n_way=args.n_way, k_spt=args.k_spt, k_qry=args.k_qry,
                       imgsz=args.imgsz, episode_num=args.train_episode_num),
            batch_size=args.task_num, shuffle=True)

        # train
        for batchidx, (spt_x, spt_y, qry_x, qry_y) in enumerate(db_train):

            spt_x, spt_y, qry_x, qry_y = spt_x.to(device), spt_y.to(device), qry_x.to(device), qry_y.to(device)

            if args.is_meta: # for meta
                loss_optim, losses_q, likelihoods_q, klds_q = net(spt_x, spt_y, qry_x, qry_y)

                global_step += 1
                if global_step % 100 == 0:

                    if args.is_vae:
                        # print(losses_q, likelihoods_q, klds_q)
                        vis.line([[losses_q[-1].item(), -likelihoods_q[-1].item(), klds_q[-1].item()]],
                                 [global_step], win='train_loss', update='append')
                    else:
                        # print(losses_q, likelihoods_q, klds_q)
                        vis.line([[losses_q[-1].item(), 0, 0]],
                                 [global_step], win='train_loss', update='append')


                    if global_step % 500 == 0:
                        if args.is_vae:
                            print(global_step)
                            print('loss_q:', torch.stack(losses_q).detach().cpu().numpy().astype(np.float16))
                            print('lkhd_q:', torch.stack(likelihoods_q).detach().cpu().numpy().astype(np.float16))
                            print('klds_q:', torch.stack(klds_q).cpu().detach().numpy().astype(np.float16))
                        else:
                            print(global_step, torch.stack(losses_q).detach().cpu().numpy().astype(np.float16))

                    if args.h_dim == 2:
                        # can not use net.decoder directly!!!
                        train_manifold = net.forward_decoder(h_manifold)
                        vis.images(train_manifold, win='train_manifold', nrow=args.h_nrow,
                                                    opts=dict(title='train_manifold:%d' % epoch))


            else: # for normal vae/ae

                loss_optim, loss_optim, likelihood, kld = net(spt_x, spt_y, qry_x, qry_y)
                optimizer.zero_grad()
                loss_optim.backward()
                optimizer.step()

                global_step += 1
                if global_step % 300 == 0:

                    print(global_step, loss_optim.item())
                    if likelihood is None:
                        vis.line([[loss_optim.item(), 0, 0]],
                                 [global_step], win='train_loss', update='append')
                    else:
                        vis.line([[loss_optim.item(), -likelihood.item(), kld.item()]],
                                 [global_step], win='train_loss', update='append')

                    if args.h_dim == 2:
                        # can not use net.decoder directly!!!
                        train_manifold = net.forward_decoder(h_manifold)
                        vis.images(train_manifold, win='train_manifold', nrow=args.h_nrow,
                                                opts=dict(title='train_manifold:%d' % epoch))




        # clustering, visualization and classification
        db_test = DataLoader(
            MnistNShot('db/mnist', training=False, n_way=args.n_way, k_spt=args.k_spt, k_qry=200,
                       imgsz=args.imgsz, episode_num=args.test_episode_num),
            batch_size=1, shuffle=True)

        for batchidx, (spt_x, spt_y, qry_x, qry_y) in enumerate(db_test):
            spt_x, spt_y, qry_x, qry_y = spt_x.to(device), spt_y.to(device), qry_x.to(device), qry_y.to(device)
            assert spt_x.size(0) == 1
            spt_x, spt_y, qry_x, qry_y = spt_x.squeeze(0), spt_y.squeeze(0), qry_x.squeeze(0), qry_y.squeeze(0)


            # we can get the representation before first update, after k update
            # and test the representation on merged(test_spt, test_qry) set
            h_spt0, h_spt1, h_qry0, h_qry1, test_manifold = net.finetuning(spt_x, spt_y, qry_x, qry_y,
                                                            args.finetuning_steps, h_manifold)

            visualh.update(h_spt0, h_spt1, h_qry0, h_qry1, spt_y, qry_y, global_step)



            acc0 = net.classify_train(h_spt0, spt_y, h_qry0, qry_y, use_h=True, train_step=args.classify_steps)
            acc1 = net.classify_train(h_spt1, spt_y, h_qry1, qry_y, use_h=True, train_step=args.classify_steps)
            print(global_step, batchidx, 'classification:\n', acc0, '\n', acc1)

            vis.line([[acc0.max(), acc1.max()]], [global_step], win='classify_acc', update='append')

            if args.h_dim == 2:
                # can not use net.decoder directly!!!
                vis.images(test_manifold, win='test_manifold', nrow=args.h_nrow,
                           opts=dict(title='test_manifold:%d' % epoch))

            break

        # # keep episode_num = batch_size for classification.
        # db_test = DataLoader(
        #     MnistNShot('db/mnist', training=False, n_way=5, k_spt=1, k_qry=15, imgsz=32, episode_num=1000),
        #     batch_size=1000, shuffle=True)
        # spt_x, spt_y, qry_x, qry_y = iter(db_test).next()
        # net.classify_train()







if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='None', help='meta/normal-fc/conv-ae/vae')
    parser.add_argument('--is_vae', action='store_true', default=False, help='ae or vae')
    parser.add_argument('--is_meta', action='store_true', default=False, help='use normal or meta version')
    parser.add_argument('--use_conv', action='store_true', default=False, help='use fc or conv')
    parser.add_argument('--task_num', type=int, default=4, help='task num, for meta and general both')
    parser.add_argument('--meta_lr', type=float, default=1e-3, help='meta lr or general lr for normal ae/vae')
    parser.add_argument('--update_num', type=int, default=5, help='update num')
    parser.add_argument('--update_lr', type=float, default=0.01, help='update lr')
    parser.add_argument('--finetuning_lr', type=float, default=0.01, help='finetuning lr, similar with update lr')
    parser.add_argument('--finetuning_steps', type=int, default=15, help='finetuning steps')
    parser.add_argument('--classify_lr', type=float, default=0.01, help='classifier lr')
    parser.add_argument('--classify_steps', type=int, default=10, help='classifier update steps')
    parser.add_argument('--n_way', type=int, default=5)
    parser.add_argument('--k_spt', type=int, default=15)
    parser.add_argument('--k_qry', type=int, default=15) # only for train-qry set
    parser.add_argument('--k_qry_test', type=int, default=200, help='in test phase')
    parser.add_argument('--imgc', type=int, default=1)
    parser.add_argument('--imgsz', type=int, default=28)
    parser.add_argument('--h_dim', type=int, default=2, help='h dim for vae. you should specify net manually for ae')
    parser.add_argument('--train_episode_num', type=int, default=5000)
    parser.add_argument('--test_episode_num', type=int, default=10)
    parser.add_argument('--epoch', type=int, default=1000, help='total epoch for training.')
    parser.add_argument('--beta', type=float, default=1., help='hyper parameters for vae')

    parser.add_argument('--h_range', type=float, default=2.0,
                        help='Range for uniformly distributed latent vector')
    parser.add_argument('--h_nrow', type=int, default=10,
                        help='number of images per row for manifold')

    args = parser.parse_args()
    main(args)
import  argparse
import  torch
from    torch import nn
from    torch.utils.data import DataLoader

from    meta import MetaAE
from    mnistNShot import MnistNShot

from    visualization import VisualH
import  numpy as np
from    matplotlib import pyplot as plt
import  visdom





def main(args):

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    device = torch.device('cuda')
    if args.is_meta:
        net = MetaAE(args)
    else:
        net =
    net.to(device)

    task_name = ''.join('meta' if args.is_meta else 'normal',
                        'conv' if args.use_conv else 'fc',
                        'vae' if args.is_vae else 'ae')

    vis = visdom.Visdom(env=task_name)
    visualh = VisualH(vis)
    global_step = 0
    vis.line([[130,120,13]], [0], win='train_loss', opts=dict(
                                                    title='train_loss',
                                                    legend=['loss', '-likelihood', 'kld'])
             )
    vis.line([[0,0]], [[0,0]], win='classify_acc', opts=dict(legend=['before', 'after'],
                                                             showlegend=True,
                                                             title='class_acc'))



    # borrowed from https://github.com/fastforwardlabs/vae-tf/blob/master/plot.py
    h_range = np.rollaxis(np.mgrid[args.h_range:-args.h_range:args.h_nrow * 1j,
                            args.h_range:-args.h_range:args.h_nrow * 1j], 0, 3)
    # [b, q_h]
    h_manifold = torch.from_numpy(h_range.reshape([-1, 2])).to(device).float()
    print('h_manifold:', h_manifold.shape)


    for epoch in range(1000):

        # 1. train
        db_train = DataLoader(
            MnistNShot('db/mnist', training=True, n_way=args.n_way, k_spt=args.k_spt, k_qry=args.k_qry,
                       imgsz=args.imgsz, episode_num=args.train_episode_num),
            batch_size=args.task_num, shuffle=True)

        # train
        for batchidx, (spt_x, spt_y, qry_x, qry_y) in enumerate(db_train):

            spt_x, spt_y, qry_x, qry_y = spt_x.to(device), spt_y.to(device), qry_x.to(device), qry_y.to(device)

            loss_optim, losses_q, likelihoods_q, klds_q = net(spt_x, spt_y, qry_x, qry_y)

            global_step += 1
            if global_step % 100 == 0:
                vis.line([[losses_q[-1].item(), -likelihoods_q[-1].item(), klds_q[-1].item()]],
                         [global_step], win='train_loss', update='append')

                # print(torch.max(spt_x), torch.min(spt_x))

                # # x_hat is composed of [spt_x_hat, qry_x_hat]
                # n = args.n_way * args.k_spt
                # x0 = spt_x.view(-1, args.imgc, args.imgsz, args.imgsz)[:n]
                # x1 = x_hat[:n]
                # comparison = torch.cat([x0, x1])
                # vis.images(comparison, nrow=args.n_way, win='train_reconstruct',
                #            opts=dict(title='train_reconstruct:%d'%global_step))


                if global_step % 500 == 0:
                    print(global_step, torch.stack(losses_q).cpu().numpy().astype(np.float16))
                    print(torch.stack(likelihoods_q).cpu().numpy().astype(np.float16))
                    print(torch.stack(klds_q).cpu().numpy().astype(np.float16))

                # can not use net.decoder directly!!!
                train_manifold = net.forward_decoder(h_manifold)
                vis.images(train_manifold, win='train_manifold', nrow=args.q_h_nrow,
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
            h_spt0, h_spt1, h_qry0, h_qry1 = net.finetuning(spt_x, spt_y, qry_x, qry_y,
                                                            update_num=args.update_num+15)

            visualh.update(h_spt0, h_spt1, h_qry0, h_qry1, spt_y, qry_y, global_step)



            acc0 = net.classify_train(h_spt0, spt_y, h_qry0, qry_y, use_h=True)
            acc1 = net.classify_train(h_spt1, spt_y, h_qry1, qry_y, use_h=True)
            print(global_step, batchidx, 'classification:\n', acc0, '\n', acc1)

            vis.line([[acc0.max(), acc1.max()]], [global_step], win='classify_acc', update='append')


            break

        # # keep episode_num = batch_size for classification.
        # db_test = DataLoader(
        #     MnistNShot('db/mnist', training=False, n_way=5, k_spt=1, k_qry=15, imgsz=32, episode_num=1000),
        #     batch_size=1000, shuffle=True)
        # spt_x, spt_y, qry_x, qry_y = iter(db_test).next()
        # net.classify_train()







if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--is_vae', action='store_true', default=False, help='ae or vae')
    parser.add_argument('--is_meta', action='store_true', default=False, help='use normal or meta version')
    parser.add_argument('--use_conv', action='store_true', default=False, help='use fc or conv')
    parser.add_argument('--task_num', type=int, default=4, help='task num, for meta and general both')
    parser.add_argument('--meta_lr', type=float, default=1e-3, help='meta lr')
    parser.add_argument('--update_num', type=int, default=5, help='update num')
    parser.add_argument('--update_lr', type=float, default=0.01, help='update lr')
    parser.add_argument('--finetunning_lr', type=float, default=0.01, help='finetunning lr, similar with update lr')
    parser.add_argument('--classify_lr', type=float, default=0.01, help='classifier lr')
    parser.add_argument('--classify_steps', type=int, default=10, help='classifier update steps')
    parser.add_argument('--n_way', type=int, default=5)
    parser.add_argument('--k_spt', type=int, default=15)
    parser.add_argument('--k_qry', type=int, default=15) # only for train-qry set
    parser.add_argument('--k_qry_test', type=int, default=200, help='in test phase')
    parser.add_argument('--imgc', type=int, default=1)
    parser.add_argument('--imgsz', type=int, default=28)
    parser.add_argument('--h_dim', type=int, default=8, help='h dim for vae. you should specify net manually for ae')
    parser.add_argument('--train_episode_num', type=int, default=10000)
    parser.add_argument('--test_episode_num', type=int, default=100)

    parser.add_argument('--h_range', type=float, default=2.0,
                        help='Range for uniformly distributed latent vector')
    parser.add_argument('--h_nrow', type=int, default=20,
                        help='number of images per row for manifold')

    args = parser.parse_args()
    main(args)
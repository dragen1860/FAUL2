import  argparse
import  torch
from    torch import optim
from    torch.utils.data import DataLoader
from    vae_fc import VAE_FC
from    mnistNShot import MnistNShot

from    visualization import VisualH
import  numpy as np
from    matplotlib import pyplot as plt
from    torchvision.utils import  make_grid, save_image
import  visdom

import  plot_utils






def main(args):

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    device = torch.device('cuda')
    net = VAE_FC(args.n_way, args.beta, args.q_h_d, args.imgc, args.imgsz)
    net.to(device)
    print(net)

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    # plt.ion()

    vis = visdom.Visdom(env='vae_fc')
    visualh = VisualH(vis)
    global_step = 0
    vis.line([[130,120,13]], [0], win='train_loss', opts={'title': 'train_loss',
                                                           'legend':['loss', '-likelihood', 'kld']})
    vis.line([[0,0]], [[0,0]], win='classify_acc', opts=dict(legend=['before', 'after'], showlegend=True,
                                                             title='class_acc'))

    # borrowed from https://github.com/fastforwardlabs/vae-tf/blob/master/plot.py
    q_h_range = np.rollaxis(np.mgrid[args.q_h_range:-args.q_h_range:args.q_h_nrow * 1j,
                            args.q_h_range:-args.q_h_range:args.q_h_nrow * 1j], 0, 3)
    # z1 = np.rollaxis(np.mgrid[1:-1:self.n_img_y * 1j, 1:-1:self.n_img_x * 1j], 0, 3)
    # z = z1**2
    # z[z1<0] *= -1
    #
    # z = z*self.z_range

    # [b, q_h]
    q_h_manifold = torch.from_numpy(q_h_range.reshape([-1, 2])).to(device).float()
    print('q_h_manifold', q_h_manifold.shape)

    # vis.surf(X=q_h_range.reshape([-1, 2]).transpose(), opts=dict(colormap='Hot'))


    for epoch in range(1000):

        # 1. train
        db_train = DataLoader(
            MnistNShot('db/mnist', training=True, n_way=args.n_way, k_spt=args.k_spt, k_qry=args.k_qry,
                       imgsz=args.imgsz, episode_num=args.train_episode_num),
            batch_size=args.task_num, shuffle=True)

        for batchidx, (spt_x, spt_y, qry_x, qry_y) in enumerate(db_train):
            # print(torch.max(spt_x), torch.min(spt_x))

            # [task_num, sptsz, 1, 28, 28]
            spt_x, spt_y, qry_x, qry_y = spt_x.to(device), spt_y.to(device), qry_x.to(device), qry_y.to(device)

            # x_hat is logits of decoders.
            loss, x_hat, likelihood, kld = net(spt_x, spt_y, qry_x, qry_y)
            # x_hat = torch.sigmoid(x_hat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            if global_step % 100 == 0:
                vis.line([[loss.item(), -likelihood.item(), kld.item()]],
                         [global_step], win='train_loss', update='append')

                # print(torch.max(spt_x), torch.min(spt_x))

                # x_hat is composed of [spt_x_hat, qry_x_hat]
                n = args.n_way * args.k_spt
                x0 = spt_x.view(-1, args.imgc, args.imgsz, args.imgsz)[:n]
                x1 = x_hat[:n]
                comparison = torch.cat([x0, x1])
                vis.images(comparison, nrow=args.n_way, win='train_reconstruct',
                           opts=dict(title='train_reconstruct:%d'%global_step))

                # comparison = make_grid(x0, nrow=n)
                # comparison = comparison.permute(1, 2, 0).cpu().detach().numpy()
                # plt.imshow(comparison)
                # plt.pause(0.001)

                if global_step % 300 == 0:
                    print(global_step, loss.item(), likelihood.item(), kld.item())

                # can not use net.decoder directly!!!
                train_manifold = net.forward_decoder(q_h_manifold)
                vis.images(train_manifold, win='train_manifold', nrow=args.q_h_nrow,
                           opts=dict(title='train_manifold:%d' % epoch))



        # clustering, visualization and classification
        db_test = DataLoader(
            MnistNShot('db/mnist', training=False, n_way=args.n_way, k_spt=args.k_spt, k_qry=args.k_qry_test,
                       imgsz=args.imgsz, episode_num=args.test_episode_num),
            batch_size=1, shuffle=True)

        for batchidx, (spt_x, spt_y, qry_x, qry_y) in enumerate(db_test):
            spt_x, spt_y, qry_x, qry_y = spt_x.to(device), spt_y.to(device), qry_x.to(device), qry_y.to(device)
            assert spt_x.size(0) == 1
            spt_x, spt_y, qry_x, qry_y = spt_x.squeeze(0), spt_y.squeeze(0), qry_x.squeeze(0), qry_y.squeeze(0)

            # we can get the representation before first update, after k update
            # and test the representation on merged(test_spt, test_qry) set
            h_spt0, h_spt1, h_qry0, h_qry1, test_manifold = net.finetuning(spt_x, spt_y, qry_x, qry_y,
                                                                        update_num=25, q_h_manifold=q_h_manifold)

            visualh.update(h_spt0, h_spt1, h_qry0, h_qry1, spt_y, qry_y, global_step)


            acc0 = net.classify_train(h_spt0, spt_y, h_qry0, qry_y, use_h=True)
            acc1 = net.classify_train(h_spt1, spt_y, h_qry1, qry_y, use_h=True)
            print(global_step, batchidx, 'classification:\n', acc0, '\n', acc1)

            vis.line([[acc0.max(), acc1.max()]], [global_step], win='classify_acc', update='append')

            # manifold
            # can not use net.decoder directly!!!
            vis.images(test_manifold, win='test_manifold', nrow=args.q_h_nrow,
                       opts=dict(title='test_manifold:%d' % epoch))


            break








if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--task_num', type=int, default=4, help='batchsz = task_num * (sptsz+qrysz)')
    parser.add_argument('--lr', type=float, default=1e-3, help='lr')
    parser.add_argument('--n_way', type=int, default=5)
    parser.add_argument('--k_spt', type=int, default=5)
    parser.add_argument('--k_qry', type=int, default=5)
    parser.add_argument('--k_qry_test', type=int, default=200, help='in test phase')
    parser.add_argument('--imgc', type=int, default=1)
    parser.add_argument('--imgsz', type=int, default=28)
    parser.add_argument('--beta', type=float, default=1.0, help='beta hyperparameters for vae')
    parser.add_argument('--q_h_d', type=int, default=2, help='convert h to q_h by linear')
    parser.add_argument('--train_episode_num', type=int, default=10000)
    parser.add_argument('--test_episode_num', type=int, default=100)


    parser.add_argument('--q_h_range', type=float, default=2.0,
                        help='Range for uniformly distributed latent vector')
    parser.add_argument('--q_h_nrow', type=int, default=20,
                        help='number of images per row for manifold')

    args = parser.parse_args()
    main(args)
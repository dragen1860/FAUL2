import  os
import  torch
from    torch.utils.data import DataLoader
import  numpy as np
from    matplotlib import pyplot as plt
import  visdom
from    datetime import datetime

from    meta import MetaAE
from    normal import AE
from    mnistNShot import MnistNShot

from    visualization import VisualH

from    sklearn import cluster, metrics
import  time



def test(args, net, device, viz=None):
    """

    :param args:
    :param net:
    :param device:
    :param visualh:
    :return:
    """
    if args.resume is None:
        print('No ckpt file specified! make sure you are training!')

    exp = args.exp + ' '

    if viz is None:
        viz = visdom.Visdom(env='test')
    visualh = VisualH(viz)

    print('Testing now...')
    output_dir = os.path.join(args.test_dir, args.exp)
    # create test_dir
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)
    # create test_dir/exp
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # clustering, visualization and classification
    db_test = DataLoader(
        MnistNShot('db/mnist', training=False, n_way=args.n_way, k_spt=args.k_spt, k_qry=args.k_qry_test,
                   imgsz=args.imgsz, episode_num=args.test_episode_num),
        batch_size=1, shuffle=True)

    viz.line([[0, 0]], [0], win='acc_on_qry01', opts=dict(title='acc_on_qry01',
                                                          legend=['h_qry0', 'h_qry1']))
    viz.line([[0, 0]], [0], win='ami_on_qry01', opts=dict(title='ami_on_qry01',
                                                          legend=['h_qry0', 'h_qry1']))
    viz.line([[0, 0]], [0], win='ars_on_qry01', opts=dict(title='ars_on_qry01',
                                                          legend=['h_qry0', 'h_qry1']))


    for ft_step in range(0, 150, 3):

        hs_spt0, hs_spt1, hs_qry0, hs_qry1, ys_spt, ys_qry = [], [], [], [], [], []
        for batchidx, (spt_x, spt_y, qry_x, qry_y) in enumerate(db_test):
            spt_x, spt_y, qry_x, qry_y = spt_x.to(device), spt_y.to(device), qry_x.to(device), qry_y.to(device)
            assert spt_x.size(0) == 1
            spt_x, spt_y, qry_x, qry_y = spt_x.squeeze(0), spt_y.squeeze(0), qry_x.squeeze(0), qry_y.squeeze(0)



            # we can get the representation before first update, after k update
            # and test the representation on merged(test_spt, test_qry) set
            h_spt0, h_spt1, h_qry0, h_qry1, _, new_net = net.finetuning(spt_x, spt_y, qry_x, qry_y,
                                                                            ft_step, None)

            hs_spt0.append(h_spt0)
            hs_spt1.append(h_spt1)
            hs_qry0.append(h_qry0)
            hs_qry1.append(h_qry1)
            ys_spt.append(spt_y)
            ys_qry.append(qry_y)

            if batchidx > 10:
                break


        h_spt0, h_spt1, h_qry0, h_qry1, y_spt, y_qry = torch.cat(hs_spt0, dim=0), torch.cat(hs_spt1, dim=0), \
                                            torch.cat(hs_qry0, dim=0), torch.cat(hs_qry1, dim=0), \
                                            torch.cat(ys_spt, dim=0), torch.cat(ys_qry, dim=0)
        print('test total size:', h_spt1.shape, h_qry1.shape, y_qry.shape)

        # we will use the acquired representation to cluster.
        # h_spt: [sptsz, h_dim]
        # h_qry: [qrysz, h_dim]
        h_qry0_np = h_qry0.detach().cpu().numpy()
        h_qry1_np = h_qry1.detach().cpu().numpy()
        y_qry_np = y_qry.detach().cpu().numpy()
        h_qry0_pred = cluster.KMeans(n_clusters=args.n_way, random_state=0).fit(h_qry0_np).labels_
        h_qry1_pred = cluster.KMeans(n_clusters=args.n_way, random_state=0).fit(h_qry1_np).labels_
        h_qry0_ami = metrics.adjusted_mutual_info_score(y_qry_np, h_qry0_pred)
        h_qry0_ars = metrics.adjusted_rand_score(y_qry_np, h_qry0_pred)
        h_qry1_ami = metrics.adjusted_mutual_info_score(y_qry_np, h_qry1_pred)
        h_qry1_ars = metrics.adjusted_rand_score(y_qry_np, h_qry1_pred)
        print(ft_step, 'ami:', h_qry0_ami, h_qry1_ami)
        print(ft_step, 'ami:', h_qry0_ars, h_qry1_ars)
        viz.line([[h_qry0_ami, h_qry1_ami]], [ft_step], win='ami_on_qry01', update='append')
        viz.line([[h_qry0_ars, h_qry1_ars]], [ft_step], win='ars_on_qry01', update='append')


        # compute contigency matrix
        # viz.heatmap(
        #     X=np.outer(np.arange(1, 6), np.arange(1, 11)),
        #     opts=dict(
        #         columnnames=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
        #         rownames=['y1', 'y2', 'y3', 'y4', 'y5'],
        #         colormap='Electric',
        #         title='heatmap'
        #     )
        # )
        # h_qry0_cm = metrics.cluster.contingency_matrix(h_qry0_pred, y_qry_np)
        # h_qry1_cm = metrics.cluster.contingency_matrix(h_qry0_pred, y_qry_np)
        # viz.heatmap(X=h_qry0_cm, win=args.exp+' h_qry0_cm', opts=dict(title=args.exp+' h_qry0_cm:%d'%batchidx,
        #                                                               colormap='Electric'))
        # viz.heatmap(X=h_qry1_cm, win=args.exp+' h_qry1_cm', opts=dict(title=args.exp+' h_qry1_cm:%d'%batchidx,
        #                                                               colormap='Electric'))



        acc0 = net.classify_train(h_spt0, y_spt, h_qry0, y_qry, use_h=True, train_step=args.classify_steps)
        acc1 = net.classify_train(h_spt1, y_spt, h_qry1, y_qry, use_h=True, train_step=args.classify_steps)
        print(ft_step, 'acc:\n', acc0, '\n', acc1)
        viz.line([[acc0[-1], acc1[-1]]], [ft_step], win='acc_on_qry01', update='append')




        for batchidx, (spt_x, spt_y, qry_x, qry_y) in enumerate(db_test):
            spt_x, spt_y, qry_x, qry_y = spt_x.to(device), spt_y.to(device), qry_x.to(device), qry_y.to(device)
            assert spt_x.size(0) == 1
            spt_x, spt_y, qry_x, qry_y = spt_x.squeeze(0), spt_y.squeeze(0), qry_x.squeeze(0), qry_y.squeeze(0)

            spt_x_hat0 = net.forward_ae(spt_x[:64])
            qry_x_hat0 = net.forward_ae(qry_x[:64])
            spt_x_hat1 = new_net.forward_ae(spt_x[:64])
            qry_x_hat1 = new_net.forward_ae(qry_x[:64])
            viz.images(qry_x[:64], nrow=8, win=exp+'qry_x', opts=dict(title=exp+'qry_x:%d'%ft_step))
            # viz.images(spt_x_hat0, nrow=8, win=exp+'spt_x_hat0', opts=dict(title=exp+'spt_x_hat0'))
            viz.images(qry_x_hat0, nrow=8, win=exp+'qry_x_hat0', opts=dict(title=exp+'qry_x_hat0:%d'%ft_step))
            # viz.images(spt_x_hat1, nrow=8, win=exp+'spt_x_hat1', opts=dict(title=exp+'spt_x_hat1'))
            viz.images(qry_x_hat1, nrow=8, win=exp+'qry_x_hat1', opts=dict(title=exp+'qry_x_hat1:%d'%ft_step))

            break

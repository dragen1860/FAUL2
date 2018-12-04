import  os
import  torch
from    torch.utils.data import DataLoader
import  numpy as np
from    matplotlib import pyplot as plt
import  visdom
from    datetime import datetime

from    meta import MetaAE
from    normal import AE
from    omniglotNShot import OmniglotNShot

from    visualization import VisualH

from    sklearn import cluster, metrics
import  time



def test_ft_steps(args, net, device, viz=None):
    """
    To plot acc/ami/ars curves with respect to every finetunning_steps, and find out the most appropriate
    finetunning steps: 35
    :param args:
    :param net:
    :param device:
    :param visualh:
    :return:
    """
    if args.resume is None:
        print('No ckpt file specified! make sure you are training!')

    exp = args.exp

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
        batch_size=1, shuffle=False)

    viz.line([[0, 0]], [0], win=exp+'acc_on_qry01', opts=dict(title=exp+'acc_on_qry01',
                                                            legend=['h_qry0', 'h_qry1'],
                                                            xlabel='finetunning_step'))
    viz.line([[0, 0]], [0], win=exp+'ami_on_qry01', opts=dict(title=exp+'ami_on_qry01',
                                                            legend=['h_qry0', 'h_qry1'],
                                                            xlabel='finetunning_step'))
    viz.line([[0, 0]], [0], win=exp+'ars_on_qry01', opts=dict(title=exp+'ars_on_qry01',
                                                            legend=['h_qry0', 'h_qry1'],
                                                            xlabel='finetunning_step'))

    for ft_step in range(1, 100, 5):

        h_qry0_ami, h_qry0_ars, h_qry1_ami, h_qry1_ars = 0, 0, 0, 0
        acc0, acc1 = [], []

        for batchidx, (spt_x, spt_y, qry_x, qry_y) in enumerate(db_test):
            spt_x, spt_y, qry_x, qry_y = spt_x.to(device), spt_y.to(device), qry_x.to(device), qry_y.to(device)
            assert spt_x.size(0) == 1
            spt_x, spt_y, qry_x, qry_y = spt_x.squeeze(0), spt_y.squeeze(0), qry_x.squeeze(0), qry_y.squeeze(0)



            # we can get the representation before first update, after k update
            # and test the representation on merged(test_spt, test_qry) set
            h_spt0, h_spt1, h_qry0, h_qry1, _, new_net = net.finetuning(spt_x, spt_y, qry_x, qry_y,
                                                                            ft_step, None)

            # visualh.update(h_spt0, h_spt1, h_qry0, h_qry1, spt_y, qry_y, batchidx)


            # we will use the acquired representation to cluster.
            # h_spt: [sptsz, h_dim]
            # h_qry: [qrysz, h_dim]
            h_qry0_np = h_qry0.detach().cpu().numpy()
            h_qry1_np = h_qry1.detach().cpu().numpy()
            qry_y_np = qry_y.detach().cpu().numpy()
            h_qry0_pred = cluster.KMeans(n_clusters=args.n_way, random_state=0).fit(h_qry0_np).labels_
            h_qry1_pred = cluster.KMeans(n_clusters=args.n_way, random_state=0).fit(h_qry1_np).labels_
            h_qry0_ami += metrics.adjusted_mutual_info_score(qry_y_np, h_qry0_pred)
            h_qry0_ars += metrics.adjusted_rand_score(qry_y_np, h_qry0_pred)
            h_qry1_ami += metrics.adjusted_mutual_info_score(qry_y_np, h_qry1_pred)
            h_qry1_ars += metrics.adjusted_rand_score(qry_y_np, h_qry1_pred)



            h_qry0_cm = metrics.cluster.contingency_matrix(h_qry0_pred, qry_y)
            h_qry1_cm = metrics.cluster.contingency_matrix(h_qry0_pred, qry_y)
            # viz.heatmap(X=h_qry0_cm, win=args.exp+' h_qry0_cm', opts=dict(title=args.exp+' h_qry0_cm:%d'%batchidx,
            #                                                               colormap='Electric'))
            # viz.heatmap(X=h_qry1_cm, win=args.exp+' h_qry1_cm', opts=dict(title=args.exp+' h_qry1_cm:%d'%batchidx,
            #                                                               colormap='Electric'))




            # return is a list of [acc_step0, acc_step1 ,...]
            acc0.append(net.classify_train(h_spt0, spt_y, h_qry0, qry_y, use_h=True, train_step=args.classify_steps))
            acc1.append(net.classify_train(h_spt1, spt_y, h_qry1, qry_y, use_h=True, train_step=args.classify_steps))




            if batchidx == 0:
                spt_x_hat0 = net.forward_ae(spt_x[:64])
                qry_x_hat0 = net.forward_ae(qry_x[:64])
                spt_x_hat1 = new_net.forward_ae(spt_x[:64])
                qry_x_hat1 = new_net.forward_ae(qry_x[:64])
                viz.images(qry_x[:64], nrow=8, win=exp+'qry_x', opts=dict(title=exp+'qry_x:%d'%ft_step))
                # viz.images(spt_x_hat0, nrow=8, win=exp+'spt_x_hat0', opts=dict(title=exp+'spt_x_hat0'))
                viz.images(qry_x_hat0, nrow=8, win=exp+'qry_x_hat0', opts=dict(title=exp+'qry_x_hat0:%d'%ft_step))
                # viz.images(spt_x_hat1, nrow=8, win=exp+'spt_x_hat1', opts=dict(title=exp+'spt_x_hat1'))
                viz.images(qry_x_hat1, nrow=8, win=exp+'qry_x_hat1', opts=dict(title=exp+'qry_x_hat1:%d'%ft_step))


            if batchidx > 2:
                break

        h_qry0_ami, h_qry0_ars, h_qry1_ami, h_qry1_ars = h_qry0_ami / (batchidx + 1), h_qry0_ars / (batchidx + 1), \
                                                         h_qry1_ami / (batchidx + 1), h_qry1_ars / (batchidx + 1)
        # [[epsode1], [episode2],...] = [N, steps] => [steps]
        acc0, acc1 = np.array(acc0).mean(axis=0), np.array(acc1).mean(axis=0)

        print(ft_step, 'ami:', h_qry0_ami, h_qry1_ami)
        print(ft_step, 'ars:', h_qry0_ars, h_qry1_ars)
        viz.line([[h_qry0_ami, h_qry1_ami]], [ft_step], win=exp+'ami_on_qry01', update='append')
        viz.line([[h_qry0_ars, h_qry1_ars]], [ft_step], win=exp+'ars_on_qry01', update='append')
        print(ft_step, 'acc:\n', acc0, '\n', acc1)
        viz.line([[acc0[-1], acc1[-1]]], [ft_step], win=exp+'acc_on_qry01', update='append')





def test_progress(args, net, device, viz=None, global_step=0):
    """
    to plot ani/ars/acc with respect to training epochs.
    :param args:
    :param net:
    :param device:
    :param viz:
    :return:
    """
    if args.resume is None:
        print('No ckpt file specified! make sure you are training!')

    exp = args.exp

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
    db_test = OmniglotNShot('db/omniglot', batchsz=1, n_way=args.n_way, k_shot=args.k_spt,
                             k_query=args.k_qry, imgsz=args.imgsz)




    h_qry0_ami, h_qry0_ars, h_qry1_ami, h_qry1_ars = 0, 0, 0, 0
    acc0, acc1 = [], []

    for batchidx in range(args.test_episode_num):
        spt_x, spt_y, qry_x, qry_y = db_test.next('test')
        spt_x, spt_y, qry_x, qry_y = torch.from_numpy(spt_x).to(device), torch.from_numpy(spt_y).to(device), \
                                     torch.from_numpy(qry_x).to(device), torch.from_numpy(qry_y).to(device)
        assert spt_x.size(0) == 1
        spt_x, spt_y, qry_x, qry_y = spt_x.squeeze(0), spt_y.squeeze(0), qry_x.squeeze(0), qry_y.squeeze(0)



        # we can get the representation before first update, after k update
        # and test the representation on merged(test_spt, test_qry) set
        h_spt0, h_spt1, h_qry0, h_qry1, _, new_net = net.finetuning(spt_x, spt_y, qry_x, qry_y,
                                                                            args.finetuning_steps, None)

        if batchidx == 0:
            visualh.update(h_spt0, h_spt1, h_qry0, h_qry1, spt_y, qry_y, global_step)


        # we will use the acquired representation to cluster.
        # h_spt: [sptsz, h_dim]
        # h_qry: [qrysz, h_dim]
        h_qry0_np = h_qry0.detach().cpu().numpy()
        h_qry1_np = h_qry1.detach().cpu().numpy()
        qry_y_np = qry_y.detach().cpu().numpy()
        h_qry0_pred = cluster.KMeans(n_clusters=args.n_way, random_state=0).fit(h_qry0_np).labels_
        h_qry1_pred = cluster.KMeans(n_clusters=args.n_way, random_state=0).fit(h_qry1_np).labels_
        h_qry0_ami += metrics.adjusted_mutual_info_score(qry_y_np, h_qry0_pred)
        h_qry0_ars += metrics.adjusted_rand_score(qry_y_np, h_qry0_pred)
        h_qry1_ami += metrics.adjusted_mutual_info_score(qry_y_np, h_qry1_pred)
        h_qry1_ars += metrics.adjusted_rand_score(qry_y_np, h_qry1_pred)



        h_qry0_cm = metrics.cluster.contingency_matrix(h_qry0_pred, qry_y)
        h_qry1_cm = metrics.cluster.contingency_matrix(h_qry0_pred, qry_y)
        # viz.heatmap(X=h_qry0_cm, win=args.exp+' h_qry0_cm', opts=dict(title=args.exp+' h_qry0_cm:%d'%batchidx,
        #                                                               colormap='Electric'))
        # viz.heatmap(X=h_qry1_cm, win=args.exp+' h_qry1_cm', opts=dict(title=args.exp+' h_qry1_cm:%d'%batchidx,
        #                                                               colormap='Electric'))




        # return is a list of [acc_step0, acc_step1 ,...]
        acc0.append(net.classify_train(h_spt0, spt_y, h_qry0, qry_y, use_h=True, train_step=args.classify_steps))
        acc1.append(net.classify_train(h_spt1, spt_y, h_qry1, qry_y, use_h=True, train_step=args.classify_steps))




        if batchidx == 0:
            spt_x_hat0 = net.forward_ae(spt_x[:64])
            qry_x_hat0 = net.forward_ae(qry_x[:64])
            spt_x_hat1 = new_net.forward_ae(spt_x[:64])
            qry_x_hat1 = new_net.forward_ae(qry_x[:64])
            viz.images(qry_x[:64], nrow=8, win=exp+'qry_x', opts=dict(title=exp+'qry_x'))
            # viz.images(spt_x_hat0, nrow=8, win=exp+'spt_x_hat0', opts=dict(title=exp+'spt_x_hat0'))
            viz.images(qry_x_hat0, nrow=8, win=exp+'qry_x_hat0', opts=dict(title=exp+'qry_x_hat0'))
            # viz.images(spt_x_hat1, nrow=8, win=exp+'spt_x_hat1', opts=dict(title=exp+'spt_x_hat1'))
            viz.images(qry_x_hat1, nrow=8, win=exp+'qry_x_hat1', opts=dict(title=exp+'qry_x_hat1'))


        if batchidx > 0:
            break


    h_qry0_ami, h_qry0_ars, h_qry1_ami, h_qry1_ars = h_qry0_ami / (batchidx + 1), h_qry0_ars / (batchidx + 1), \
                                                     h_qry1_ami / (batchidx + 1), h_qry1_ars / (batchidx + 1)
    # [[epsode1], [episode2],...] = [N, steps] => [steps]
    acc0, acc1 = np.array(acc0).mean(axis=0), np.array(acc1).mean(axis=0)

    print('ami:', h_qry0_ami, h_qry1_ami)
    print('ars:', h_qry0_ars, h_qry1_ars)
    viz.line([[h_qry0_ami, h_qry1_ami]], [global_step], win=exp+'ami_on_qry01', update='append')
    viz.line([[h_qry0_ars, h_qry1_ars]], [global_step], win=exp+'ars_on_qry01', update='append')
    print('acc:\n', acc0, '\n', acc1)
    viz.line([[acc0[-1], acc1[-1]]], [global_step], win=exp+'acc_on_qry01', update='append')

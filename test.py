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




def test(args, net, device):
    """

    :param args:
    :param net:
    :param device:
    :param visualh:
    :return:
    """
    if args.resume is None:
        raise NotImplementedError

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
        MnistNShot('db/mnist', training=False, n_way=args.n_way, k_spt=args.k_spt, k_qry=200,
                   imgsz=args.imgsz, episode_num=args.test_episode_num),
        batch_size=1, shuffle=True)

    for batchidx, (spt_x, spt_y, qry_x, qry_y) in enumerate(db_test):
        spt_x, spt_y, qry_x, qry_y = spt_x.to(device), spt_y.to(device), qry_x.to(device), qry_y.to(device)
        assert spt_x.size(0) == 1
        spt_x, spt_y, qry_x, qry_y = spt_x.squeeze(0), spt_y.squeeze(0), qry_x.squeeze(0), qry_y.squeeze(0)

        # we can get the representation before first update, after k update
        # and test the representation on merged(test_spt, test_qry) set
        h_spt0, h_spt1, h_qry0, h_qry1, _ = net.finetuning(spt_x, spt_y, qry_x, qry_y,
                                                                       args.finetuning_steps, None)



        visualh.update(h_spt0, h_spt1, h_qry0, h_qry1, spt_y, qry_y, batchidx)


        # we will use the acquired representation to cluster.
        # h_spt: [sptsz, h_dim]
        # h_qry: [qrysz, h_dim]
        h_qry0_np = h_qry0.detach().cpu().numpy()
        h_qry1_np = h_qry1.detach().cpu().numpy()
        qry_y_np = qry_y.detach().cpu().numpy()
        h_qry0_pred = cluster.KMeans(n_clusters=args.n_way, random_state=0).fit(h_qry0_np).labels_
        h_qry1_pred = cluster.KMeans(n_clusters=args.n_way, random_state=0).fit(h_qry1_np).labels_
        h_qry0_ami = metrics.adjusted_mutual_info_score(qry_y_np, h_qry0_pred)
        h_qry0_ars = metrics.adjusted_rand_score(qry_y_np, h_qry0_pred)
        h_qry1_ami = metrics.adjusted_mutual_info_score(qry_y_np, h_qry1_pred)
        h_qry1_ars = metrics.adjusted_rand_score(qry_y_np, h_qry1_pred)
        print(batchidx, 'ami:', h_qry0_ami, h_qry1_ami)
        print(batchidx, 'ami:', h_qry0_ars, h_qry1_ars)


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
        h_qry0_cm = metrics.cluster.contingency_matrix(h_qry0_pred, qry_y)
        h_qry1_cm = metrics.cluster.contingency_matrix(h_qry0_pred, qry_y)
        viz.heatmap(X=h_qry0_cm, win=args.exp+' h_qry0_cm', opts=dict(title=args.exp+' h_qry0_cm:%d'%batchidx,
                                                                      colormap='Electric'))
        viz.heatmap(X=h_qry1_cm, win=args.exp+' h_qry1_cm', opts=dict(title=args.exp+' h_qry1_cm:%d'%batchidx,
                                                                      colormap='Electric'))



        acc0 = net.classify_train(h_spt0, spt_y, h_qry0, qry_y, use_h=True, train_step=args.classify_steps)
        acc1 = net.classify_train(h_spt1, spt_y, h_qry1, qry_y, use_h=True, train_step=args.classify_steps)
        print(batchidx, 'acc:\n', acc0, '\n', acc1)








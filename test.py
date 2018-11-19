import  argparse
import  os
import  torch
from    torch import optim
from    torch.utils.data import DataLoader
import  numpy as np
from    matplotlib import pyplot as plt
import  visdom
from    datetime import datetime

from    meta import MetaAE
from    normal import AE
from    mnistNShot import MnistNShot

from    visualization import VisualH

from    sklearn import manifold, datasets, decomposition




def test(args, net, device, visualh):
    """

    :param args:
    :param net:
    :param device:
    :param visualh:
    :return:
    """
    if args.resume is None:
        raise NotImplementedError

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

        # we will use the acquired representation to cluster.

        visualh.update(h_spt0, h_spt1, h_qry0, h_qry1, spt_y, qry_y, global_step)

        acc0 = net.classify_train(h_spt0, spt_y, h_qry0, qry_y, use_h=True, train_step=args.classify_steps)
        acc1 = net.classify_train(h_spt1, spt_y, h_qry1, qry_y, use_h=True, train_step=args.classify_steps)
        print(global_step, batchidx, 'classification:\n', acc0, '\n', acc1)

        vis.line([[acc0.max(), acc1.max()]], [global_step], win='classify_acc', update='append')



        break



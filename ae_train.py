import  argparse
import  torch
from    torch import nn
from    torch.utils.data import DataLoader
from    ae import AE
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
    net = AE(args.n_way, args.imgc, args.imgsz)
    net.to(device)


    vis = visdom.Visdom()
    visualh = VisualH(vis)
    global_step = 0
    vis.line([0.25], [0], win='train_loss', opts={'title': 'train_loss'})
    vis.line([[0,0]], [[0,0]], win='classify_acc', opts=dict(legend=['before', 'after'], showlegend=True,
                                                             title='class_acc'))

    for epoch in range(1000):

        # 1. train
        db_train = DataLoader(
            MnistNShot('db/mnist', training=True, n_way=args.n_way, k_spt=args.k_spt, k_qry=args.k_qry,
                       imgsz=args.imgsz, episode_num=args.train_episode_num),
            batch_size=args.task_num, shuffle=True)

        for batchidx, (spt_x, spt_y, qry_x, qry_y) in enumerate(db_train):

            spt_x, spt_y, qry_x, qry_y = spt_x.to(device), spt_y.to(device), qry_x.to(device), qry_y.to(device)

            loss, x_hat = net(spt_x, spt_y, qry_x, qry_y)

            global_step += 1
            if global_step % 20 == 0:
                vis.line([loss.item()], [global_step], win='train_loss', update='append')

                if global_step % 200 == 0:
                    print(global_step, loss.item())

        # clustering, visualization and classification
        db_test = DataLoader(
            MnistNShot('db/mnist', training=False, n_way=args.n_way, k_spt=args.k_spt, k_qry=45,
                       imgsz=args.imgsz, episode_num=args.test_episode_num),
            batch_size=1, shuffle=True)

        for batchidx, (spt_x, spt_y, qry_x, qry_y) in enumerate(db_test):
            spt_x, spt_y, qry_x, qry_y = spt_x.to(device), spt_y.to(device), qry_x.to(device), qry_y.to(device)
            assert spt_x.size(0) == 1
            spt_x, spt_y, qry_x, qry_y = spt_x.squeeze(0), spt_y.squeeze(0), qry_x.squeeze(0), qry_y.squeeze(0)

            # we can get the representation before first update, after k update
            # and test the representation on merged(test_spt, test_qry) set
            h_spt0, h_spt1, h_qry0, h_qry1 = net.finetuning(spt_x, spt_y, qry_x, qry_y, update_num=25)

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
    parser.add_argument('--task_num', type=int, default=40, help='batchsz = task_num * (sptsz+qrysz)')
    parser.add_argument('--meta_lr', type=float, default=1e-3, help='meta lr')
    parser.add_argument('--update_num', type=int, default=5, help='update num')
    parser.add_argument('--update_lr', type=float, default=0.01, help='update lr')
    parser.add_argument('--n_way', type=int, default=5)
    parser.add_argument('--k_spt', type=int, default=1)
    parser.add_argument('--k_qry', type=int, default=15)
    parser.add_argument('--imgc', type=int, default=1)
    parser.add_argument('--imgsz', type=int, default=28)
    parser.add_argument('--train_episode_num', type=int, default=5000)
    parser.add_argument('--test_episode_num', type=int, default=100)
    args = parser.parse_args()
    main(args)
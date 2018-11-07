import  argparse
import  torch
from    torch import nn
from    torch.utils.data import DataLoader
from    metaae import MetaLearner
from    mnistNShot import MnistNShot













def main(args):

    device = torch.device('cuda')
    net = MetaLearner(args.n_way, args.k_spt, args.k_qry, args.task_num, args.update_num, args.meta_lr, args.update_lr)
    net.to(device)

    db_test = DataLoader(MnistNShot('db/mnist', training=False, n_way=5, k_spt=1, k_qry=15, imgsz=32, episode_num=1000),
                          batch_size= args.task_num, shuffle=True)



    for epoch in range(100):

        db_train = DataLoader(
            MnistNShot('db/mnist', training=True, n_way=5, k_spt=1, k_qry=15, imgsz=32, episode_num=10000),
            batch_size=args.task_num, shuffle=True)

        for batchidx, (spt_x, spt_y, qry_x, qry_y) in enumerate(db_train):

            spt_x, spt_y, qry_x, qry_y = spt_x.to(device), spt_y.to(device), qry_x.to(device), qry_y.to(device)

            net(spt_x, spt_y, qry_x, qry_y)








if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--task_num', type=int, default=8, help='task num')
    parser.add_argument('--meta_lr', type=float, default=1e-3, help='meta lr')
    parser.add_argument('--update_num', type=int, default=5, help='update num')
    parser.add_argument('--update_lr', type=float, default=0.05, help='update lr')
    parser.add_argument('--n_way', type=int, default=5)
    parser.add_argument('--k_spt', type=int, default=1)
    parser.add_argument('--k_qry', type=int, default=15)
    parser.add_argument('--imgsz', type=int, default=32)
    parser.add_argument('--h_d', type=int, default=4)
    parser.add_argument('--h_c', type=int, default=4)
    args = parser.parse_args()
    main(args)
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
import  test

def update_args(args):

    if args.exp is None:
        task_name = ''.join(['meta' if args.is_meta else 'normal', '-',
                             'conv' if args.use_conv else 'fc', '-',
                             'vae' if args.is_vae else 'ae'])
        args.exp = task_name
        print('Not set exp flags, we generate it as:', task_name)
        # once exp is not specified, we use default individual flags.
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
        args.finetuning_lr = 0.1
        args.finetuning_steps = 15


    elif exp == 'meta-fc-vae':
        args.is_vae = True
        args.is_meta = True
        args.use_conv = False
        args.finetuning_lr = 0.1
        args.finetuning_steps = 15



    elif exp == 'normal-fc-ae':
        args.is_vae = False
        args.is_meta = False
        args.use_conv = False
        args.finetuning_lr = 0.1 # distinct from meta, this should be smaller
        args.finetuning_steps = 15


    elif exp == 'normal-fc-vae':
        args.is_vae = True
        args.beta = 1.0
        args.is_meta = False
        args.use_conv = False
        args.finetuning_lr = 0.1 # distinct from meta, this should be smaller
        args.finetuning_steps = 15


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
        # model_parameters = filter(lambda p: p.requires_grad, net.learner.parameters())
        # params = sum([np.prod(p.size()) for p in model_parameters])
        # print('Total params:', params)
        tmp = filter(lambda x: x.requires_grad, net.learner.parameters())
        num = sum(map(lambda x:np.prod(x.shape), tmp))
        print('Total trainable variables:', num)
    else:
        net = AE(args, use_logits=True)
        optimizer = optim.Adam(list(net.encoder.parameters()) + list(net.decoder.parameters()),
                               lr=args.meta_lr)

        tmp = filter(lambda x: x.requires_grad, list(net.encoder.parameters())+list(net.decoder.parameters()))
        num = sum(map(lambda x:np.prod(x.shape), tmp))
        print('Total trainable variables:', num)

    net.to(device)
    print(net)


    print('='*15,'Experiment:', args.exp, '='*15)
    print(args)

    if args.h_dim == 2:
        # borrowed from https://github.com/fastforwardlabs/vae-tf/blob/master/plot.py
        h_range = np.rollaxis(np.mgrid[args.h_range:-args.h_range:args.h_nrow * 1j,
                                args.h_range:-args.h_range:args.h_nrow * 1j], 0, 3)
        # [b, q_h]
        h_manifold = torch.from_numpy(h_range.reshape([-1, 2])).to(device).float()
        print('h_manifold:', h_manifold.shape)
    else:
        h_manifold = None


    # try to resume from ckpt.mdl file
    epoch_start = 0
    global_step = 0
    if args.resume is not None:
        # ckpt/normal-fc-vae_640_2018-11-20_09:58:58.mdl
        mdl_file = args.resume
        epoch_start = int(mdl_file.split('_')[-3])
        net.load_state_dict(torch.load(mdl_file))
        global_step = int(epoch_start * args.train_episode_num / args.task_num)
        print('Resume from:', args.resume, 'epoch:', epoch_start, 'global_step:', global_step)
    else:
        print('Training/test from scratch...')


    if args.test:
        assert args.resume is not None
        test.test_ft_steps(args, net, device)
        return




    vis = visdom.Visdom(env=args.exp)
    visualh = VisualH(vis)
    vis.line([[0,0,0]], [0], win='train_loss', opts=dict(
                                                    title='train_loss',
                                                    legend=['loss', '-lklh', 'kld'],
                                                    xlabel='global_step')
             )

    # for test_progress
    vis.line([[0, 0]], [0], win=args.exp+'acc_on_qry01', opts=dict(title=args.exp+'acc_on_qry01',
                                                          legend=['h_qry0', 'h_qry1'],
                                                            xlabel='global_step'))
    vis.line([[0, 0]], [0], win=args.exp+'ami_on_qry01', opts=dict(title=args.exp+'ami_on_qry01',
                                                          legend=['h_qry0', 'h_qry1'],
                                                                   xlabel='global_step'))
    vis.line([[0, 0]], [0], win=args.exp+'ars_on_qry01', opts=dict(title=args.exp+'ars_on_qry01',
                                                          legend=['h_qry0', 'h_qry1'],
                                                                   xlabel='global_step'))

    for epoch in range(epoch_start, args.epoch):

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
                if global_step % 300 == 0:

                    if args.is_vae:
                        # print(losses_q, likelihoods_q, klds_q)
                        vis.line([[losses_q[-1].item(), -likelihoods_q[-1].item(), klds_q[-1].item()]],
                                 [global_step], win='train_loss', update='append')
                        print(epoch, global_step)
                        print('loss_q:', torch.stack(losses_q).detach().cpu().numpy().astype(np.float16))
                        print('lkhd_q:', torch.stack(likelihoods_q).detach().cpu().numpy().astype(np.float16))
                        print('klds_q:', torch.stack(klds_q).cpu().detach().numpy().astype(np.float16))
                    else:
                        # print(losses_q, likelihoods_q, klds_q)
                        vis.line([[losses_q[-1].item(), 0, 0]],
                                 [global_step], win='train_loss', update='append')
                        print(epoch, global_step, torch.stack(losses_q).detach().cpu().numpy().astype(np.float16))




            else: # for normal vae/ae

                loss_optim, _, likelihood, kld = net(spt_x, spt_y, qry_x, qry_y)
                optimizer.zero_grad()
                loss_optim.backward()
                torch.nn.utils.clip_grad_norm_(list(net.encoder.parameters())+list(net.decoder.parameters()), 10)
                optimizer.step()

                global_step += 1
                if global_step % 300 == 0:

                    print(epoch, global_step, loss_optim.item())
                    if not args.is_vae:
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

        if epoch % 5 == 0:
            test.test_progress(args, net, device, vis, global_step)


        # save checkpoint.
        if epoch % 20 == 0:
            date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            mdl_file = os.path.join(args.ckpt_dir, args.exp + '_%d'%epoch  + '_' + date_str + '.mdl')
            torch.save(net.state_dict(), mdl_file)
            print('Saved into ckpt file:', mdl_file)


    # save checkpoint.
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    mdl_file = os.path.join(args.ckpt_dir, args.exp + '_%d'%args.epoch  + '_' + date_str + '.mdl')
    torch.save(net.state_dict(), mdl_file)
    print('Saved Last state ckpt file:', mdl_file)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='None', help='meta/normal-fc/conv-ae/vae')
    parser.add_argument('--test', action='store_true', help='set to test')
    parser.add_argument('--is_vae', action='store_true', default=False, help='ae or vae')
    parser.add_argument('--is_meta', action='store_true', default=False, help='use normal or meta version')
    parser.add_argument('--use_conv', action='store_true', default=False, help='use fc or conv')
    parser.add_argument('--task_num', type=int, default=4, help='task num, for meta and general both')
    parser.add_argument('--meta_lr', type=float, default=1e-3, help='meta lr or general lr for normal ae/vae')
    parser.add_argument('--update_num', type=int, default=5, help='update num')
    parser.add_argument('--update_lr', type=float, default=0.5, help='update lr')
    parser.add_argument('--finetuning_lr', type=float, default=0.1, help='finetuning lr, similar with update lr')
    parser.add_argument('--finetuning_steps', type=int, default=15, help='finetuning steps')
    parser.add_argument('--classify_lr', type=float, default=0.01, help='classifier lr')
    parser.add_argument('--classify_steps', type=int, default=50, help='classifier update steps')
    parser.add_argument('--n_way', type=int, default=5)
    parser.add_argument('--k_spt', type=int, default=1)
    parser.add_argument('--k_qry', type=int, default=20) # only for train-qry set
    parser.add_argument('--k_qry_test', type=int, default=200, help='in test phase')
    parser.add_argument('--imgc', type=int, default=1)
    parser.add_argument('--imgsz', type=int, default=28)
    parser.add_argument('--h_dim', type=int, default=2, help='h dim for vae. you should specify net manually for ae')
    parser.add_argument('--train_episode_num', type=int, default=5000)
    parser.add_argument('--test_episode_num', type=int, default=100)
    parser.add_argument('--ckpt_dir', type=str, default='ckpt', help='checkpoint save directory')
    parser.add_argument('--test_dir', type=str, default='results', help='directory to save test results images and figures')
    parser.add_argument('--resume', type=str, default=None, help='--resume ckpt.mdl file.')
    parser.add_argument('--epoch', type=int, default=300, help='total epoch for training.')
    parser.add_argument('--beta', type=float, default=1., help='hyper parameters for vae')


    parser.add_argument('--fc_hidden', type=int, default=128, help='784=>fc_hidden=>')
    parser.add_argument('--conv_ch', type=int, default=16, help='conv channels units')

    parser.add_argument('--h_range', type=float, default=2.0,
                        help='Range for uniformly distributed latent vector')
    parser.add_argument('--h_nrow', type=int, default=10,
                        help='number of images per row for manifold')

    args = parser.parse_args()
    main(args)
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
from    omniglotNShot import OmniglotNShot

from    visualization import VisualH
import  omni_test as test



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
        args.finetuning_lr = 0.01
        args.finetuning_steps = 15
    elif exp == 'meta-conv-vae':
        args.is_vae = True
        args.is_meta = True
        args.use_conv = True
        args.finetuning_lr = 0.01
        args.finetuning_steps = 15
    elif exp == 'normal-conv-ae':
        args.is_vae = False
        args.is_meta = False
        args.use_conv = True
        args.finetuning_lr = 0.01
        args.finetuning_steps = 15
    elif exp == 'normal-conv-vae':
        args.is_vae = True
        args.is_meta = False
        args.use_conv = True
        args.finetuning_lr = 0.01
        args.finetuning_steps = 15



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
        # args.finetuning_lr = 0.1 # distinct from meta, this should be smaller
        # args.finetuning_steps = 15


    elif exp == 'normal-fc-vae':
        args.is_vae = True
        args.beta = 1.0
        args.is_meta = False
        args.use_conv = False
        # args.finetuning_lr = 0.1 # distinct from meta, this should be smaller
        # args.finetuning_steps = 15


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
        num = sum(map(lambda x: np.prod(x.shape), tmp))
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
    if args.resume is not None:
        # ckpt/normal-fc-vae_640_2018-11-20_09:58:58.mdl
        mdl_file = args.resume
        epoch_start = int(mdl_file.split('_')[-3])
        net.load_state_dict(torch.load(mdl_file))
        print('Resume from:', args.resume, 'epoch/batches:', epoch_start)
    else:
        print('Training/test from scratch...')


    if args.test:
        assert args.resume is not None
        test.test_ft_steps(args, net, device)
        return




    vis = visdom.Visdom(env=args.exp)
    visualh = VisualH(vis)
    vis.line([[0,0,0]], [epoch_start], win=args.exp+'train_loss', opts=dict(
                                                    title=args.exp+'train_qloss',
                                                    legend=['loss', '-lklh', 'kld'],
                                                    xlabel='global_step'))

    # for test_progress
    vis.line([[0, 0]], [epoch_start], win=args.exp+'acc_on_qry01', opts=dict(title=args.exp+'acc_on_qry01',
                                                          legend=['h_qry0', 'h_qry1'],
                                                            xlabel='global_step'))
    vis.line([[0, 0]], [epoch_start], win=args.exp+'ami_on_qry01', opts=dict(title=args.exp+'ami_on_qry01',
                                                          legend=['h_qry0', 'h_qry1'],
                                                                   xlabel='global_step'))
    vis.line([[0, 0]], [epoch_start], win=args.exp+'ars_on_qry01', opts=dict(title=args.exp+'ars_on_qry01',
                                                          legend=['h_qry0', 'h_qry1'],
                                                                   xlabel='global_step'))

    # 1. train
    db_train = OmniglotNShot('db/omniglot', batchsz=args.task_num, n_way=args.n_way, k_shot=args.k_spt,
                             k_query=args.k_qry, imgsz=args.imgsz)

    # epoch = batch number here.
    for epoch in range(epoch_start, args.train_episode_num):
        spt_x, spt_y, qry_x, qry_y = db_train.next()
        spt_x, spt_y, qry_x, qry_y = torch.from_numpy(spt_x).to(device), torch.from_numpy(spt_y).to(device), \
                                     torch.from_numpy(qry_x).to(device), torch.from_numpy(qry_y).to(device)


        if args.is_meta: # for meta
            loss_optim, losses_q, likelihoods_q, klds_q = net(spt_x, spt_y, qry_x, qry_y)


            if epoch % 300 == 0:

                if args.is_vae:
                    # print(losses_q, likelihoods_q, klds_q)
                    vis.line([[losses_q[-1].item(), -likelihoods_q[-1].item(), klds_q[-1].item()]],
                             [epoch], win=args.exp+'train_loss', update='append')
                    print(epoch)
                    print('loss_q:', torch.stack(losses_q).detach().cpu().numpy().astype(np.float16))
                    print('lkhd_q:', torch.stack(likelihoods_q).detach().cpu().numpy().astype(np.float16))
                    print('klds_q:', torch.stack(klds_q).cpu().detach().numpy().astype(np.float16))
                else:
                    # print(losses_q, likelihoods_q, klds_q)
                    vis.line([[losses_q[-1].item(), 0, 0]],
                             [epoch], win=args.exp+'train_loss', update='append')
                    print(epoch, torch.stack(losses_q).detach().cpu().numpy().astype(np.float16))




        else: # for normal vae/ae

            loss_optim, _, likelihood, kld = net(spt_x, spt_y, qry_x, qry_y)
            optimizer.zero_grad()
            loss_optim.backward()
            torch.nn.utils.clip_grad_norm_(list(net.encoder.parameters())+list(net.decoder.parameters()), 10)
            optimizer.step()

            if epoch % 300 == 0:

                print(epoch, loss_optim.item())
                if not args.is_vae:
                    vis.line([[loss_optim.item(), 0, 0]],
                             [epoch], win='train_loss', update='append')
                else:
                    vis.line([[loss_optim.item(), -likelihood.item(), kld.item()]],
                             [epoch], win='train_loss', update='append')






        if epoch % 3000 == 0:
            # [qrysz, 1, 64, 64] => [qrysz, 1, 64, 64]
            x_hat = net.forward_ae(qry_x[0])
            vis.images(x_hat, nrow=args.k_qry, win='train_x_hat', opts=dict(title='train_x_hat'))
            test.test_progress(args, net, device, vis, epoch)


        # save checkpoint.
        if epoch % 10000 == 0:
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
    parser.add_argument('--update_lr', type=float, default=0.1, help='update lr')
    parser.add_argument('--finetuning_lr', type=float, default=0.1, help='finetuning lr, similar with update lr')
    parser.add_argument('--finetuning_steps', type=int, default=5, help='finetuning steps')
    parser.add_argument('--classify_lr', type=float, default=0.02, help='classifier lr')
    parser.add_argument('--classify_steps', type=int, default=50, help='classifier update steps')
    parser.add_argument('--n_way', type=int, default=5)
    parser.add_argument('--k_spt', type=int, default=1)
    parser.add_argument('--k_qry', type=int, default=15) # only for train-qry set
    parser.add_argument('--k_qry_test', type=int, default=15, help='in test phase')
    parser.add_argument('--imgc', type=int, default=1)
    parser.add_argument('--imgsz', type=int, default=64)
    parser.add_argument('--h_dim', type=int, default=20, help='h dim for vae. you should specify net manually for ae')
    parser.add_argument('--train_episode_num', type=int, default=5000000)
    parser.add_argument('--test_episode_num', type=int, default=100)
    parser.add_argument('--ckpt_dir', type=str, default='ckpt', help='checkpoint save directory')
    parser.add_argument('--test_dir', type=str, default='results', help='directory to save test results images and figures')
    parser.add_argument('--resume', type=str, default=None, help='--resume ckpt.mdl file.')
    parser.add_argument('--beta', type=float, default=1., help='hyper parameters for vae')


    parser.add_argument('--fc_hidden', type=int, default=128, help='784=>fc_hidden=>')
    parser.add_argument('--conv_ch', type=int, default=64, help='conv channels units')

    parser.add_argument('--h_range', type=float, default=2.0,
                        help='Range for uniformly distributed latent vector')
    parser.add_argument('--h_nrow', type=int, default=10,
                        help='number of images per row for manifold')

    args = parser.parse_args()
    main(args)
import  torch
from    torch import nn
from    torch.nn import functional as F
import  numpy as np



class AELearner(nn.Module):
    """
    This meta-based network is appropriate for ae and vae, supporting conv and fc.
    """

    def __init__(self, config, imgc, imgsz, is_vae, beta):
        """

        :param config: network config file, type:list of (string, list)
        :param imgc: 1 or 3
        :param imgsz:  28 or 64
        :param is_vae: auto-encoder or variational ae
        :param beta: beta for vae.
        """
        super(AELearner, self).__init__()


        self.config = config
        self.is_vae = is_vae
        self.beta = beta
        # will be determined by config file.
        self.use_logits = False

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()
        # running_mean and running_var
        self.vars_bn = nn.ParameterList()

        for i, (name, param) in enumerate(self.config):
            if name is 'conv2d':
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name is 'convt2d':
                # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_in, ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[1])))

            elif name is 'linear':
                # [ch_out, ch_in]
                w = nn.Parameter(torch.ones(*param))
                # gain=1 according to cbfinn's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name is 'bn':
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])


            elif name is 'usigma_layer':
                assert self.is_vae
                # w1 + b1 => mean
                # w2 + b2 => sigma
                # [ch_out, ch_in]
                w1 = nn.Parameter(torch.ones(*param))
                w2 = nn.Parameter(torch.ones(*param))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w1)
                torch.nn.init.kaiming_normal_(w2)
                self.vars.append(w1)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
                self.vars.append(w2)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name is 'hidden':
                # to separate the variable from encoder.
                self.hidden_var_idx = len(self.vars)
                self.hidden_var_bn_idx = len(self.vars_bn)
                # to separate network config
                self.hidden_config_idx = i
                print('hidden_var_idx:', self.hidden_var_idx, 'hidden_var_bn_idx:', self.hidden_var_bn_idx)

            elif name is 'use_logits':
                self.use_logits = True
                print('Use sigmoid with logits, make sure no implicit sigmoid in network config!')

            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
                continue
            else:
                raise NotImplementedError



        if self.use_logits:
            # target: [b, 1, 28, 28]
            self.criteon = nn.BCEWithLogitsLoss(reduction='elementwise_mean')
            # self.criteon = nn.MSELoss(reduction='sum')
        else:
            self.criteon = nn.BCELoss(reduction='elementwise_mean')
            # self.criteon = nn.MSELoss(reduction='elementwise_mean')


        # TODO: can not use torch.Tensor since it will update running_mean and running_var to np.inf
        h = self.forward_encoder(torch.randn(2, imgc, imgsz, imgsz))
        self.h_shape = h.shape[1:] # for printing network structure
        # return with x, loss, likelihood, kld
        out, _, _, _ = self.forward(torch.randn(2, imgc, imgsz, imgsz))
        print('Meta','VAE' if is_vae else 'AE', end=' ')
        print([2, imgc, imgsz, imgsz], '>:', h.shape, ':<', list(out.shape))


    def extra_repr(self):
        info = ''

        for name, param in self.config:
            if name is 'conv2d':
                tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'convt2d':
                tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[0], param[1], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'linear':
                tmp = 'linear:(in:%d, out:%d)'%(param[1], param[0])
                info += tmp + '\n'
            elif name is 'usigma_layer':
                tmp = 'usigma_layer:(in:%d, mean:%d, sigma^2:%d)'%(param[1], param[0], param[0])
                info += tmp + '\n'
            elif name is 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)'%(param[0])
                info += tmp + '\n'

            elif name is 'hidden':
                tmp = name + ': >>' + str(list(self.h_shape)) + '<<'
                info += tmp + '\n'

            elif name is 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name is 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError

        return info



    def forward(self, x, vars=None, update_bn_statistics=True):
        """
        This function can be called by finetunning, however, in finetunning, we dont wish to update
        running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weiths.
        :param x: [b, 1, 28, 28]
        :param vars:
        :param update_bn_statistics: set False to not update
        :return: x, loss, likelihood, kld
        """

        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0
        input = x

        for name, param in self.config:
            if name is 'conv2d':
                w, b = vars[idx:(idx + 2)]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name is 'convt2d':
                w, b = vars[idx:(idx + 2)]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name is 'linear':
                w, b = vars[idx:(idx + 2)]
                x = F.linear(x, w, b)
                idx += 2
                # print('forward:', idx, x.norm().item())
            elif name is 'bn':
                w, b = vars[idx:(idx + 2)]
                running_mean, running_var = self.vars_bn[bn_idx:(bn_idx+2)]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=update_bn_statistics)
                idx += 2
                bn_idx += 2

            elif name is 'usigma_layer':
                w1, b1 = vars[idx:(idx + 2)]
                w2, b2 = vars[idx+2:(idx + 4)]
                # [b, h_dim]
                x1 = F.linear(x, w1, b1)
                # [b, h_dim]
                x2 = F.linear(x, w2, b2)
                # [b, 2*h_dim]
                x = torch.cat([x1, x2], dim=1)
                idx += 4
            elif name is 'flatten':
                x = x.view(x.size(0), -1)
            elif name is 'reshape':
                # [b, 8] => [b, 2, 2, 2]
                x = x.view(x.size(0), *param)
            elif name is 'relu':
                x = F.relu(x, inplace=param[0])
            elif name is 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
            elif name is 'tanh':
                x = F.tanh(x)
            elif name is 'sigmoid':
                x = torch.sigmoid(x)
            elif name is 'hidden':
                if self.is_vae:
                    # convert from h to q_h
                    # [b, 2*q_h_d]
                    assert len(x.shape)==2
                    # splitting current x into mu and sigma
                    q_mu, q_sigma = x.chunk(2, dim=1)
                    # reparametrize trick
                    q_h = q_mu + q_sigma * torch.randn_like(q_sigma)
                    x = q_h
                else:
                    continue

            elif name is 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])
            elif name is 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])
            elif name is 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])
            elif name is 'use_logits':
                continue
            else:
                raise NotImplementedError

        # make sure variable is used properly
        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)


        if self.is_vae:
            # assert not torch.isnan(x).any()

            # likelihood is the negative loss.
            likelihood = -self.criteon(x, input)

            # see Appendix B from VAE paper:
            # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
            # https://arxiv.org/abs/1312.6114
            # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            kld = 0.5 * torch.sum(
                                torch.pow(q_mu, 2) +
                                torch.pow(q_sigma, 2) -
                                torch.log(1e-8 + torch.pow(q_sigma, 2)) - 1
                            ) / np.prod(input.shape)
            kld = self.beta * kld
            elbo = likelihood - kld
            loss = - elbo

            if self.use_logits:
                x = torch.sigmoid(x)

            return x, loss, likelihood, kld

        else:
            loss = self.criteon(x, input)
            # print(loss, input.shape)

            if self.use_logits:
                x = torch.sigmoid(x)

            return x, loss, None, None

    def forward_encoder(self, x, vars=None):
        """
        forward till hidden layer
        :param x:
        :param vars:
        :return:
        """
        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0

        for name, param in self.config:
            if name is 'conv2d':
                w, b = vars[idx:(idx + 2)]
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
            elif name is 'convt2d':
                w, b = vars[idx:(idx + 2)]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name is 'linear':
                w, b = vars[idx:(idx + 2)]
                x = F.linear(x, w, b)
                idx += 2
            elif name is 'bn':
                w, b = vars[idx:(idx + 2)]
                # TODO: can not be written as running_mean, running_var = self.vars_bn[bn_idx:(bn_idx+1)]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=False)
                idx += 2
                bn_idx += 2
            elif name is 'usigma_layer':
                w1, b1 = vars[idx:(idx + 2)]
                w2, b2 = vars[idx + 2:(idx + 4)]
                # [b, h_dim]
                x1 = F.linear(x, w1, b1)
                # [b, h_dim]
                x2 = F.linear(x, w2, b2)
                # [b, 2*h_dim]
                x = torch.cat([x1, x2], dim=1)
                idx += 4
            elif name is 'flatten':
                x = x.view(x.size(0), -1)
            elif name is 'reshape':
                # [b, 8] => [b, 2, 2, 2]
                x = x.view(x.size(0), *param)
            elif name is 'relu':
                x = F.relu(x, inplace=param[0])
            elif name is 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
            elif name is 'tanh':
                x = F.tanh(x)
            elif name is 'sigmoid':
                x = torch.sigmoid(x)
            elif name is 'hidden':

                if self.is_vae:
                    # convert from h to q_h
                    # [b, 2*q_h_d]
                    assert len(x.shape)==2
                    # splitting current x into mu and sigma
                    q_mu, q_sigma = x.chunk(2, dim=1)
                    # reparametrize trick
                    q_h = q_mu + q_sigma * torch.randn_like(q_sigma)
                    x = q_h


                break

            elif name is 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])

            elif name is 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])
            elif name is 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])
            elif name is 'use_logits':
                raise NotImplementedError
            else:
                raise NotImplementedError

        assert idx == self.hidden_var_idx
        assert bn_idx == self.hidden_var_bn_idx

        return x



    def forward_decoder(self, h, vars=None):
        """
        forward after hidden layer
        :param h:
        :param vars:
        :return:
        """
        if vars is None:
            vars = self.vars


        # get decoder network config
        decoder_config = self.config[self.hidden_config_idx+1:]
        idx = self.hidden_var_idx
        bn_idx = self.hidden_var_bn_idx

        x = h
        for name, param in decoder_config:
            if name is 'conv2d':
                w, b = vars[idx:(idx + 2)]
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
            elif name is 'convt2d':
                w, b = vars[idx:(idx + 2)]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name is 'linear':
                w, b = vars[idx:(idx + 2)]
                x = F.linear(x, w, b)
                idx += 2
            elif name is 'bn':
                w, b = vars[idx:(idx + 2)]
                # TODO: can not be written as running_mean, running_var = self.vars_bn[bn_idx:(bn_idx+1)]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=False)
                idx += 2
                bn_idx += 2
            elif name is 'usigma_layer':
                w1, b1 = vars[idx:(idx + 2)]
                w2, b2 = vars[idx+2:(idx + 4)]
                # [b, h_dim]
                x1 = F.linear(x, w1, b1)
                # [b, h_dim]
                x2 = F.linear(x, w2, b2)
                # [b, 2*h_dim]
                x = torch.cat([x1, x2], dim=1)
                idx += 4
            elif name is 'flatten':
                x = x.view(x.size(0), -1)
            elif name is 'reshape':
                # [b, 8] => [b, 2, 2, 2]
                x = x.view(x.size(0), *param)
            elif name is 'hidden':
                raise NotImplementedError
            elif name is 'tanh':
                x = F.tanh(x)
            elif name is 'sigmoid':
                x = torch.sigmoid(x)
            elif name is 'relu':
                x = F.relu(x, inplace=param[0])
            elif name is 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
            elif name is 'hidden':
                raise NotImplementedError
            elif name is 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])
            elif name is 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])
            elif name is 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])
            elif name is 'use_logits':
                continue
            else:
                raise NotImplementedError

        # print(self.hidden_var_idx, idx, len(self.vars))

        assert  idx == len(vars)
        assert  bn_idx == len(self.vars_bn)

        if self.use_logits:
            x = torch.sigmoid(x)

        return x


    def forward_ae(self, x, vars=None):
        """

        :param x:
        :param vars:
        :return:
        """
        if vars is None:
            vars = self.vars
        h = self.forward_encoder(x, vars)
        x_hat = self.forward_decoder(h, vars)

        # w,b = self.vars[2], self.vars[3]
        # running_mean, running_var = self.vars_bn[0], self.vars_bn[1]
        # print(w.norm().item(), b.norm().item(), running_mean.norm().item(), running_var.norm().item())

        return x_hat

    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars
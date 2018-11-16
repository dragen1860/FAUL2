import  torch
from    torch import nn
from    torch.nn import functional as F




class AELearner(nn.Module):
    """
    This meta-based network is appropriate for ae and vae, supporting conv and fc.
    """

    def __init__(self, config, imgc, imgsz, is_vae, beta):
        """

        :param config: network config file
        :param imgc:
        :param imgsz:
        :param is_vae: auto-encoder or variational ae
        """
        super(AELearner, self).__init__()


        self.config = config
        self.is_vae = is_vae
        self.beta = beta
        self.use_logits = False

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()

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
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name is 'hidden':
                # to separate the variable from encoder.
                self.hidden_var_idx = len(self.vars)
                # to separate network config
                self.hidden_config_idx = i
                print('hidden_vars:', self.hidden_var_idx, 'hidden_config:', self.hidden_config_idx)

            elif name is 'use_logits':
                self.use_logits = True
                print('will use logits on last layer, make sure no implicit sigmoid in network config!')
            elif name in ['tanh', 'relu', 'hidden', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
                continue
            else:
                raise NotImplementedError

        h = self.forward_encoder(torch.Tensor(2, imgc, imgsz, imgsz))
        # return with x, loss, likelihood, kld
        out, _, _, _ = self.forward(torch.Tensor(2, imgc, imgsz, imgsz))
        print('Meta','VAE' if is_vae else 'AE', end=' ')
        print([2, imgc, imgsz, imgsz], '>:', h.shape, ':<', list(out.shape))



    def extra_repr(self):
        info = ''

        for name, param in self.config:
            if name is 'conv2d':
                tmp = 'conv2d:(ch_out:%d, ch_in:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[0], param[1], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'convt2d':
                tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[0], param[1], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'linear':
                tmp = 'linear:(in:%d, out:%d)'%(param[1], param[0])
                info += tmp + '\n'
            elif name is 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)'%(param[0])
                info += tmp + '\n'

            elif name is 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name is 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name in ['flatten', 'tanh', 'relu', 'hidden', 'upsample', 'reshape', 'sigmoid', 'use_logits']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError

        return info



    def forward(self, x, vars=None):
        """

        :param x: [b, 1, 28, 28]
        :param vars:
        :return:
        """

        if vars is None:
            vars = self.vars

        idx = 0
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
        assert idx == len(self.vars)



        if self.is_vae:
            assert not torch.isnan(x).any()
            if self.use_logits:
                likelihood = -F.binary_cross_entropy_with_logits(x, input, reduction='sum') / input.size(0)
            else:
                likelihood = -F.binary_cross_entropy(x, input, reduction='sum') / input.size(0)

            # see Appendix B from VAE paper:
            # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
            # https://arxiv.org/abs/1312.6114
            # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            kld = 0.5 * torch.sum(
                torch.pow(q_mu, 2) +
                torch.pow(q_sigma, 2) -
                torch.log(1e-8 + torch.pow(q_sigma, 2)) - 1
            ).sum() / input.size(0)
            elbo = likelihood - self.beta * kld
            loss = - elbo

            if self.use_logits:
                x = torch.sigmoid(x)

            return x, loss, likelihood, kld

        else:
            if self.use_logits:
                loss = F.binary_cross_entropy_with_logits(x, input, reduction='sum')/ input.size(0)
            else:
                loss = F.binary_cross_entropy(x, input, reduction='sum')/ input.size(0)

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

        hidden_loc = 0
        for name, param in self.config:
            if name is not 'hidden':
                 hidden_loc += 1
            else:
                break

        if hidden_loc == len(self.config):
            raise NotImplementedError

        # get decoder network config
        encoder_config = self.config[:hidden_loc]

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

        assert  idx == len(self.vars)

        if self.use_logits:
            x = torch.sigmoid(x)

        return x

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
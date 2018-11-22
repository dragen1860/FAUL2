import  os, random
import  torch
from    torch.utils.data import Dataset, DataLoader

from    torchvision import transforms
import  numpy as np
from    torchvision import datasets


class MnistNShot(Dataset):
    """
    NOTICE: meta-learning is different from general supervised learning, especially the concept of batchsz and setsz.
    batch: contains several sets
    sets: conains n_way * k_shot for meta-train set, n_way * n_query for meta-test set.
    """

    def __init__(self, root, training, n_way, k_spt, k_qry, imgsz, episode_num=10000):
        """

        :param root: data path
        :param n_way:
        :param k_spt:
        :param k_qry:
        :param imgsz: resize image to imgsz
        :param episode_num: total episode num, although we can generate infinite number of sets, but we treat it
                            as finite.
        """
        self.root = root
        self.n_way = n_way  # n-way
        self.k_spt = k_spt  # k-shot
        self.k_qry = k_qry  # for evaluation
        self.sptsz = self.n_way * self.k_spt  # num of samples per set
        self.qrysz = self.n_way * self.k_qry  # number of samples per set for evaluation
        self.imgsz = imgsz  # resize to
        self.episode_num = episode_num
        # print('shuffle %s DB: %d-way, %d-shot, %d-query, resize:%d episode num:%d' %
        #       ('train' if training else 'test', n_way, k_spt, k_qry, imgsz, episode_num))

        if training:
            self.transform = transforms.Compose([
                                                 # lambda x: Image.open(x).convert('RGB'),
                                                 transforms.Resize((self.imgsz, self.imgsz)),
                                                 # transforms.RandomHorizontalFlip(),
                                                 # transforms.RandomRotation(5),
                                                 transforms.ToTensor(),
                                                 # transforms.Normalize((0.5,), (1.,))
                                                 ])
        else:
            self.transform = transforms.Compose([
                                                 # lambda x: Image.open(x).convert('RGB'),
                                                 transforms.Resize((self.imgsz, self.imgsz)),
                                                 transforms.ToTensor(),
                                                 # transforms.Normalize((0.5,), (1.,))
                                                 ])

        db1 = DataLoader(datasets.MNIST(root, train=True, transform=self.transform, download=True),
                        batch_size=60000, num_workers=1, pin_memory=True)
        data1 = iter(db1).next()
        db2 = DataLoader(datasets.MNIST(root, train=False, transform=self.transform, download=True),
                        batch_size=10000, num_workers=1, pin_memory=True)
        data2 = iter(db2).next()
        # torch.Size([60000, 1, 32, 32]) torch.Size([60000])
        # torch.Size([10000, 1, 32, 32]) torch.Size([10000])
        # print(data1[0].shape, data1[1].shape)
        # print(data2[0].shape, data2[1].shape)
        x = torch.cat([data1[0], data2[0]], dim=0)
        label = torch.cat([data1[1], data2[1]], dim=0)
        # print(x.shape, label.shape)

        # every element is a list including all images belonging to same category
        # TODO: CAN NOT write as [[]]*10, since data[0] and data[1] will use same reference
        data = [[], [], [], [], [], [], [], [], [], []]
        for (x_, label_) in zip(x, label):
            data[label_].append(x_)

        num_len = list(map(lambda x: len(x), data))
        # print('mnist class num:', num_len)

        if training:
            self.mode_numbers = np.array([0, 1, 2, 3, 4, 5])
        else:
            self.mode_numbers = np.array([6, 7, 8, 9])


        # sample specific categories
        self.data = []
        for l in self.mode_numbers:
            # put data with label in the end of self.data
            # print(data[l][0].shape) # [1, 32, 32]
            # [ [l_img1, l_img2,...], [...]] => [tensor_l1, tensor_l2, ... ]
            imgs = torch.stack(data[l], dim=0)
            # print(imgs.shape) [6903, 1, 32, 32]
            self.data.append(imgs)
        # relative indexing
        # [l, l2, l3]
        self.table = np.arange(len(self.mode_numbers))

        # we need enough categories to sample
        assert n_way <= len(self.mode_numbers)

        self.cls_num = len(self.data)

        # create all episode ahead
        self.create_episodes(self.episode_num)




    def create_episodes(self, episode_num):
        """
        create episode for meta-learning.
        :param episode_num:
        :return:
        """
        self.episodes = []

        for b in range(episode_num):  # for each batch
            # 1.select n_way classes randomly
            selected_cls = np.random.choice(self.cls_num, self.n_way, False)  # no duplicate
            np.random.shuffle(selected_cls)
            spt_x, spt_y, qry_x, qry_y = [], [], [], []
            for cls in selected_cls:
                img_num_per_cls = self.k_spt + self.k_qry
                # 2. select k_shot + k_query for each class
                selected_imgs_idx = np.random.choice(len(self.data[cls]), img_num_per_cls,
                                                     False if img_num_per_cls < len(self.data[cls]) else True)
                #    select label | select setsz | select spt or qry = [k_spt/k_qry, 1, 32, 32]
                spt_x.append(self.data[cls][selected_imgs_idx][:self.k_spt])
                # cls: relative index label
                # self.table[cls]: global index label
                spt_y.extend([cls]*self.k_spt)
                qry_x.append(self.data[cls][selected_imgs_idx][self.k_spt:])
                qry_y.extend([cls]*self.k_qry)
            # list of [k_spt, 1, 32, 32] => [sptsz, 1, 32, 32]
            spt_x = torch.cat(spt_x, dim=0)
            # [sptsz]
            spt_y = torch.from_numpy(np.array(spt_y)).long()
            # list of [k_qry, 1, 32, 32] => [qrysz, 1, 32, 32]
            qry_x = torch.cat(qry_x, dim=0)
            # [qrysz]
            qry_y = torch.from_numpy(np.array(qry_y)).long()
            # shuffle inside set
            perm = torch.randperm(self.sptsz)
            spt_x = spt_x[perm]
            spt_y = spt_y[perm]
            perm = torch.randperm(self.qrysz)
            qry_x = qry_x[perm]
            qry_y = qry_y[perm]


            assert not torch.isnan(spt_x).any()
            assert not torch.isnan(spt_y).any()
            assert not torch.isnan(qry_x).any()
            assert not torch.isnan(qry_y).any()

            # spt_x: tensor[sptsz, 1, 32, 32]
            # spt_y: tensor[sptsz]
            # qry_x: tensor[qrysz, 1, 32, 32]
            # qry_y: tensor[qrysz]
            self.episodes.append([spt_x, spt_y, qry_x, qry_y])


    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= episode_num-1
        :param index:
        :return:
        """
        return self.episodes[index]


    def __len__(self):
        # as we have built up to episode_num of episode
        return self.episode_num


if __name__ == '__main__':
    # the following episode is to view one set of images via tensorboard.
    from    torchvision.utils import make_grid
    from    matplotlib import pyplot as plt
    import  visdom
    import  time

    plt.ion()
    vis = visdom.Visdom()

    db = MnistNShot('db/mnist', training=True, n_way=5, k_spt=1, k_qry=15, imgsz=32)

    for i, set_ in enumerate(db):
        # support_x: [k_shot*n_way, 3, 84, 84]
        support_x, support_y, query_x, query_y = set_
        print(support_x.shape, support_y.shape, query_x.shape, query_y.shape)

        # de-normalize
        support_x += 0.5
        query_x += 0.5

        support_x_all = make_grid(support_x, nrow=5)
        query_x_all = make_grid(query_x, nrow=5)

        plt.figure(1)
        # [b, 1, 32, 32] => [b, 32, 32, 1]
        plt.imshow(support_x_all.permute(1, 2, 0).numpy())
        plt.pause(0.5)
        plt.figure(2)
        plt.imshow(query_x_all.permute(1, 2, 0).numpy())
        plt.pause(0.5)

        print(support_y)
        print(query_y[:5], query_y[5:10], query_y[-5:])

        vis.images(support_x, nrow=5, win='spt_x', opts={'title':'spt_x'})
        vis.images(query_x, nrow=5, win='qry_x', opts={'title':'qry_x'})


        time.sleep(15)

    vis.close()

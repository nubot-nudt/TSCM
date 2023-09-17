# %%
import importlib
import os
import pickle
import shutil
from os.path import dirname, exists, join
import h5py
import faiss
import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import json
import torch.optim as optim
from torchsummary import summary

os.sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import time
from options import FixRandom
from utils import cal_recall, light_log, schedule_device
import importlib
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from scipy.io import loadmat
from collections import namedtuple

dbStruct = namedtuple('dbStruct',
                      ['whichSet', 'dataset', 'dbImage', 'utmDb', 'qImage', 'utmQ', 'numDb', 'numQ', 'posDistThr',
                       'posDistSqThr', 'nonTrivPosDistSqThr'])

class CKD_loss(nn.Module):
    def __init__(self, margin) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, embs_a, embs_p, embs_n, mu_tea_a, mu_tea_p, mu_tea_n):  # (1, D)
        SaTp = torch.norm(embs_a - mu_tea_p, p=2).pow(2)
        SpTa = torch.norm(embs_p - mu_tea_a, p=2).pow(2)

        SaTn = torch.norm(embs_a - mu_tea_n, p=2).pow(2)
        SnTa = torch.norm(embs_n - mu_tea_a, p=2).pow(2)

        SaTa = torch.norm(embs_a - mu_tea_a, p=2).pow(2)
        SpTp = torch.norm(embs_p - mu_tea_p, p=2).pow(2)
        SnTn = torch.norm(embs_n - mu_tea_n, p=2).pow(2)
        dis_D = SpTp + SnTn
        # dis_D =SaTp+SpTa+SaTa+SpTp+SnTn
        # dis_D=SaTa+SpTp+SnTn
        loss = 0.5 * (torch.clamp(self.margin + dis_D, min=0).pow(2))

        return loss


class Trainer:
    def __init__(self, options) -> None:

        self.opt = options

        # r variables
        self.step = 0
        self.epoch = 0
        self.current_lr = 0
        self.best_recalls = [0, 0, 0]

        # seed
        fix_random = FixRandom(self.opt.seed)
        self.seed_worker = fix_random.seed_worker()
        self.time_stamp = datetime.now().strftime('%m%d_%H%M%S')

        # set device
        if self.opt.phase == 'train_tea':
            self.opt.cGPU = schedule_device()
        if self.opt.cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run with --nocuda :(")
        torch.cuda.set_device(self.opt.cGPU)
        self.device = torch.device("cuda")
        print('{}:{}{}'.format('device', self.device, torch.cuda.current_device()))

        # CKD_loss
        self.CKD_loss = CKD_loss(margin=torch.tensor(self.opt.margin, device=self.device))
        # make model
        if self.opt.phase == 'train_tea':
            self.model, self.optimizer, self.scheduler, self.criterion = self.make_model()
        elif self.opt.phase == 'train_stu':
            self.teacher_net, self.student_net, self.optimizer, self.scheduler, self.criterion = self.make_model()
            self.model = self.teacher_net
        elif self.opt.phase in ['test_tea', 'test_stu']:
            self.model = self.make_model()
        else:
            raise Exception('Undefined phase :(')

        # make folders
        self.make_folders()
        # make dataset
        self.make_dataset()
        # online logs
        if self.opt.phase in ['train_tea', 'train_stu']:
            wandb.init(project="STUN", config=vars(self.opt),
                       name=f"{self.opt.loss}_{self.opt.phase}_{self.time_stamp}")

    def make_folders(self):
        ''' create folders to store tensorboard files and a copy of networks files
        '''
        if self.opt.phase in ['train_tea', 'train_stu']:
            self.opt.runsPath = join(self.opt.logsPath, f"{self.opt.loss}_{self.opt.phase}_{self.time_stamp}")
            if not os.path.exists(join(self.opt.runsPath, 'models')):
                os.makedirs(join(self.opt.runsPath, 'models'))

            if not os.path.exists(join(self.opt.runsPath, 'transformed')):
                os.makedirs(join(self.opt.runsPath, 'transformed'))

            for file in [__file__, 'datasets/{}.py'.format(self.opt.dataset), 'networks/{}.py'.format(self.opt.net)]:
                shutil.copyfile(file, os.path.join(self.opt.runsPath, 'models', file.split('/')[-1]))

            with open(join(self.opt.runsPath, 'flags.json'), 'w') as f:
                f.write(json.dumps({k: v for k, v in vars(self.opt).items()}, indent=''))

    def make_dataset(self):
        ''' make dataset
        '''
        if self.opt.phase in ['train_tea', 'train_stu']:
            assert os.path.exists(f'datasets/{self.opt.dataset}.py'), 'Cannot find ' + f'{self.opt.dataset}.py :('
            self.dataset = importlib.import_module('datasets.' + self.opt.dataset)
        elif self.opt.phase in ['test_tea', 'test_stu']:
            self.dataset = importlib.import_module('tmp.models.{}'.format(self.opt.dataset))

        # for emb cache
        self.whole_train_set = self.dataset.get_whole_training_set(self.opt)
        self.whole_training_data_loader = DataLoader(dataset=self.whole_train_set, num_workers=self.opt.threads,
                                                     batch_size=self.opt.cacheBatchSize, shuffle=False,
                                                     pin_memory=self.opt.cuda, worker_init_fn=self.seed_worker)
        self.whole_val_set = self.dataset.get_whole_val_set(self.opt)
        self.whole_val_data_loader = DataLoader(dataset=self.whole_val_set, num_workers=self.opt.threads,
                                                batch_size=self.opt.cacheBatchSize, shuffle=False,
                                                pin_memory=self.opt.cuda, worker_init_fn=self.seed_worker)
        self.whole_test_set = self.dataset.get_whole_test_set(self.opt)
        self.whole_test_data_loader = DataLoader(dataset=self.whole_test_set, num_workers=self.opt.threads,
                                                 batch_size=self.opt.cacheBatchSize, shuffle=False,
                                                 pin_memory=self.opt.cuda, worker_init_fn=self.seed_worker)
        # for train tuples
        self.train_set = self.dataset.get_training_query_set(self.opt, self.opt.margin)
        self.training_data_loader = DataLoader(dataset=self.train_set, num_workers=8, batch_size=self.opt.batchSize,
                                                   shuffle=True, collate_fn=self.dataset.collate_fn,
                                                   worker_init_fn=self.seed_worker)
        print('{}:{}, {}:{}, {}:{}, {}:{}, {}:{}'.format('dataset', self.opt.dataset, 'database',
                                                         self.whole_train_set.dbStruct.numDb, 'train_set',
                                                         self.whole_train_set.dbStruct.numQ, 'val_set',
                                                         self.whole_val_set.dbStruct.numQ, 'test_set',
                                                         self.whole_test_set.dbStruct.numQ))
        print('{}:{}, {}:{}'.format('cache_bs', self.opt.cacheBatchSize, 'tuple_bs', self.opt.batchSize))

    def make_model(self):
        '''build model
        '''
        if self.opt.phase == 'train_tea':
            # build teacher net
            assert os.path.exists(f'networks/{self.opt.net}.py'), 'Cannot find ' + f'{self.opt.net}.py :('
            network = importlib.import_module('networks.' + self.opt.net)
            model = network.deliver_model(self.opt, 'tea')
            model = model.to(self.device)
            outputs = model(torch.rand((2, 3, self.opt.height, self.opt.width), device=self.device))
            self.opt.output_dim = \
            model(torch.rand((2, 3, self.opt.height, self.opt.width), device=self.device))[0].shape[-1]
            self.opt.sigma_dim = \
            model(torch.rand((2, 3, self.opt.height, self.opt.width), device=self.device))[1].shape[-1]  # place holder
        elif self.opt.phase == 'train_stu':  # load teacher net
            assert self.opt.resume != '', 'You need to define the teacher/resume path :('
            if exists('tmp'):
                shutil.rmtree('tmp')
            os.mkdir('tmp')
            shutil.copytree(join(dirname(self.opt.resume), 'models'), join('tmp', 'models'))
            network = importlib.import_module(f'tmp.models.{self.opt.net}')
            model_tea = network.deliver_model(self.opt, 'tea').to(self.device)
            checkpoint = torch.load(self.opt.resume)
            model_tea.load_state_dict(checkpoint['state_dict'])
            # build student net
            assert os.path.exists(f'networks/{self.opt.net}.py'), 'Cannot find ' + f'{self.opt.net}.py :('
            network = importlib.import_module('networks.' + self.opt.net)
            model = network.deliver_model(self.opt, 'stu').to(self.device)
            #checkpointS = torch.load('logs/tri_train_stu_0820_220921/ckpt_e_56.pth.tar')
            #model.load_state_dict(checkpointS['state_dict'])
            self.opt.output_dim = \
            model(torch.rand((2, 3, self.opt.height, self.opt.width), device=self.device))[0].shape[-1]
            self.opt.sigma_dim = \
            model(torch.rand((2, 3, self.opt.height, self.opt.width), device=self.device))[1].shape[-1]
        elif self.opt.phase in ['test_tea', 'test_stu']:
            # load teacher or student net
            assert self.opt.resume != '', 'You need to define a teacher/resume path :('
            if exists('tmp'):
                shutil.rmtree('tmp')
            os.mkdir('tmp')
            shutil.copytree(join(dirname(self.opt.resume), 'models'), join('tmp', 'models'))
            network = importlib.import_module('tmp.models.{}'.format(self.opt.net))
            model = network.deliver_model(self.opt, self.opt.phase[-3:]).to(self.device)
            checkpoint = torch.load(self.opt.resume)
            model.load_state_dict(checkpoint['state_dict'])

        print('{}:{}, {}:{}, {}:{}'.format(model.id, self.opt.net, 'loss', self.opt.loss, 'mu_dim', self.opt.output_dim,
                                           'sigma_dim', self.opt.sigma_dim if self.opt.phase[-3:] == 'stu' else '-'))

        if self.opt.phase in ['train_tea', 'train_stu']:
            # optimizer
            if self.opt.optim == 'adam':
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), self.opt.lr,
                                       weight_decay=self.opt.weightDecay)
                scheduler = optim.lr_scheduler.ExponentialLR(optimizer, self.opt.lrGamma, last_epoch=-1, verbose=False)
            elif self.opt.optim == 'sgd':
                optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=self.opt.lr,
                                      momentum=self.opt.momentum, weight_decay=self.opt.weightDecay)
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.opt.lrStep, gamma=self.opt.lrGamma)
            else:
                raise NameError('Undefined optimizer :(')

            # loss function
            criterion = nn.TripletMarginLoss(margin=self.opt.margin, p=2, reduction='sum').to(self.device)

        if self.opt.nGPU > 1:
            model = nn.DataParallel(model)

        if self.opt.phase == 'train_tea':
            return model, optimizer, scheduler, criterion
        elif self.opt.phase == 'train_stu':
            return model_tea, model, optimizer, scheduler, criterion
        elif self.opt.phase in ['test_tea', 'test_stu']:
            return model
        else:
            raise NameError('Undefined phase :(')

    def build_embedding_cache(self):
        '''build embedding cache, such that we can find the corresponding (p) and (n) with respect to (a) in embedding space
        '''
        self.train_set.cache = os.path.join(self.opt.runsPath, self.train_set.whichSet + '_feat_cache.hdf5')
        with h5py.File(self.train_set.cache, mode='w') as h5:
            h5feat = h5.create_dataset("features", [len(self.whole_train_set), self.opt.output_dim], dtype=np.float32)
            with torch.no_grad():
                for iteration, (input, indices) in enumerate(tqdm(self.whole_training_data_loader), 1):
                    input = input.to(self.device)  # torch.Size([32, 3, 154, 154]) ([32, 5, 3, 200, 200])
                    emb, _ = self.model(input)
                    h5feat[indices.detach().numpy(), :] = emb.detach().cpu().numpy()
                    del input, emb

    def build_embedding_cache_stu(self):
        '''build embedding cache, such that we can find the corresponding (p) and (n) with respect to (a) in embedding space
        '''
        self.train_set.cache = os.path.join(self.opt.runsPath, self.train_set.whichSet + '_feat_cache.hdf5')
        with h5py.File(self.train_set.cache, mode='w') as h5:
            h5feat = h5.create_dataset("features", [len(self.whole_train_set), self.opt.output_dim], dtype=np.float32)
            with torch.no_grad():
                for iteration, (input, indices) in enumerate(tqdm(self.whole_training_data_loader), 1):
                    input = input.to(self.device)  # torch.Size([32, 3, 154, 154]) ([32, 5, 3, 200, 200])
                    emb, _ = self.student_net(input)
                    h5feat[indices.detach().numpy(), :] = emb.detach().cpu().numpy()
                    del input, emb

    def process_batch(self, batch_inputs):
        '''
        process a batch of input
        '''
        anchor, positives, negatives, neg_counts, indices = batch_inputs

        # in case we get an empty batch
        if anchor is None:
            return None, None

        # some reshaping to put query, pos, negs in a single (N, 3, H, W) tensor, where N = batchSize * (nQuery + nPos + n_neg)
        B = anchor.shape[0]  # ([8, 1, 3, 200, 200])
        n_neg = torch.sum(neg_counts)  # tensor(80) = torch.sum(torch.Size([8]))

        input = torch.cat([anchor, positives, negatives])  # ([B, C, H, 200])

        input = input.to(self.device)  # ([96, 1, C, H, W])
        embs, vars = self.model(input)  # ([96, D])

        tuple_loss = 0
        # Standard triplet loss (via PyTorch library)
        if self.opt.loss == 'tri':
            embs_a, embs_p, embs_n = torch.split(embs, [B, B, n_neg])
            for i, neg_count in enumerate(neg_counts):
                for n in range(neg_count):
                    negIx = (torch.sum(neg_counts[:i]) + n).item()
                    tuple_loss += self.criterion(embs_a[i:i + 1], embs_p[i:i + 1], embs_n[negIx:negIx + 1])
            tuple_loss /= n_neg.float().to(self.device)

        del input, embs, embs_a, embs_p, embs_n
        del anchor, positives, negatives

        return tuple_loss, n_neg

    def process_batch_stu(self, batch_inputs):
        '''
        process a batch of input
        '''
        anchor, positives, negatives, neg_counts, indices = batch_inputs

        # in case we get an empty batch
        if anchor is None:
            return None, None

        # some reshaping to put query, pos, negs in a single (N, 3, H, W) tensor, where N = batchSize * (nQuery + nPos + n_neg)
        B = anchor.shape[0]  # ([8, 1, 3, 200, 200])
        n_neg = torch.sum(neg_counts)  # tensor(80) = torch.sum(torch.Size([8]))
        input = torch.cat([anchor, positives, negatives])  # ([B, C, H, 200])

        input = input.to(self.device)  # ([96, 1, C, H, W])
        embs, vars = self.student_net(input)  # ([96, D])

        anchor = anchor.to(self.device)
        with torch.no_grad():
            mu_tea, _ = self.teacher_net(input)  # ([B, D])
        # mu_stu, log_sigma_sq = self.student_net(anchor)  # ([B, D]), ([B, D])

        tuple_loss = 0
        loss = 0
        CKDloss = 0

        # Standard triplet loss (via PyTorch library)
        if self.opt.loss == 'tri':
            embs_a, embs_p, embs_n = torch.split(embs, [B, B, n_neg])
            vars_a, vars_p, vars_n = torch.split(vars, [B, B, n_neg])
            mu_tea_a, mu_tea_p, mu_tea_n = torch.split(mu_tea, [B, B, n_neg])
            for i, neg_count in enumerate(neg_counts):
                for n in range(neg_count):
                    negIx = (torch.sum(neg_counts[:i]) + n).item()
                    tuple_loss += self.criterion(embs_a[i:i + 1], embs_p[i:i + 1], embs_n[negIx:negIx + 1])
                    CKDloss += self.CKD_loss(embs_a[i:i + 1], embs_p[i:i + 1], embs_n[negIx:negIx + 1],
                                                 mu_tea_a[i:i + 1], mu_tea_p[i:i + 1], mu_tea_n[negIx:negIx + 1])

            tuple_loss /= n_neg.float().to(self.device)
            CKDloss /= n_neg.float().to(self.device)

        del input, embs, embs_a, embs_p, embs_n
        del anchor, positives, negatives

        return loss, n_neg

    def train(self):
        not_improved = 0
        for epoch in range(self.opt.nEpochs):
            self.epoch = epoch
            self.current_lr = self.optimizer.state_dict()['param_groups'][0]['lr']

            # build embedding cache
            if self.epoch % self.opt.cacheRefreshEvery == 0:
                self.model.eval()
                self.build_embedding_cache()
                self.model.train()

            # train
            tuple_loss_sum = 0
            for _, batch_inputs in enumerate(tqdm(self.training_data_loader)):
                self.step += 1

                self.optimizer.zero_grad()
                tuple_loss, n_neg = self.process_batch(batch_inputs)
                if tuple_loss is None:
                    continue
                tuple_loss.backward()
                self.optimizer.step()
                tuple_loss_sum += tuple_loss.item()

                if self.step % 10 == 0:
                    wandb.log({'train_tuple_loss': tuple_loss.item()}, step=self.step)
                    wandb.log({'train_batch_num_neg': n_neg}, step=self.step)

            n_batches = len(self.training_data_loader)
            wandb.log({'train_avg_tuple_loss': tuple_loss_sum / n_batches}, step=self.step)
            torch.cuda.empty_cache()
            self.scheduler.step()

            # val every x epochs
            if (self.epoch % self.opt.evalEvery) == 0:
                recalls = self.val(self.model)
                if recalls[0] > self.best_recalls[0]:
                    self.best_recalls = recalls
                    not_improved = 0
                else:
                    not_improved += self.opt.evalEvery
                # light log
                vars_to_log = [
                    'e={:>2d},'.format(self.epoch),
                    'lr={:>.8f},'.format(self.current_lr),
                    'tl={:>.4f},'.format(tuple_loss_sum / n_batches),
                    'r@1/5/10={:.2f}/{:.2f}/{:.2f}'.format(recalls[0], recalls[1], recalls[2]),
                    '\n' if not_improved else ' *\n',
                ]
                light_log(self.opt.runsPath, vars_to_log)
            else:
                recalls = None
            self.save_model(self.model, is_best=not not_improved)

            # stop when not improving for a period
            if self.opt.phase == 'train_tea':
                if self.opt.patience > 0 and not_improved > self.opt.patience:
                    print('terminated because performance has not improve for', self.opt.patience, 'epochs')
                    break

        self.save_model(self.model, is_best=False)
        print('best r@1/5/10={:.2f}/{:.2f}/{:.2f}'.format(self.best_recalls[0], self.best_recalls[1],
                                                          self.best_recalls[2]))

        return self.best_recalls

    def train_student(self):
        not_improved = 0
        for epoch in range(self.opt.nEpochs):
            self.epoch = epoch
            self.current_lr = self.optimizer.state_dict()['param_groups'][0]['lr']

            # build embedding cache
            if self.epoch % self.opt.cacheRefreshEvery == 0:
                self.student_net.eval()
                self.build_embedding_cache()
                self.student_net.train()
                # train
                tuple_loss_sum = 0
                for _, batch_inputs in enumerate(tqdm(self.training_data_loader)):
                    self.step += 1

                    self.optimizer.zero_grad()
                    tuple_loss, n_neg = self.process_batch_stu(batch_inputs)
                    if tuple_loss is None:
                        continue
                    tuple_loss.backward()
                    self.optimizer.step()
                    tuple_loss_sum += tuple_loss.item()
                    loss_sum = tuple_loss_sum
                    if self.step % 10 == 0:
                        wandb.log({'train_tuple_loss': tuple_loss.item()}, step=self.step)
                        wandb.log({'train_batch_num_neg': n_neg}, step=self.step)

                n_batches = len(self.training_data_loader)
                wandb.log({'train_avg_tuple_loss': tuple_loss_sum / n_batches}, step=self.step)
                wandb.log({'student/epoch_loss': loss_sum / n_batches}, step=self.step)
                torch.cuda.empty_cache()
                self.scheduler.step()

            # val
            if (self.epoch % self.opt.evalEvery) == 0:
                recalls = self.val(self.student_net)
                if recalls[0] > self.best_recalls[0]:
                    self.best_recalls = recalls
                    not_improved = 0
                else:
                    not_improved += self.opt.evalEvery

                light_log(self.opt.runsPath, [
                    f'e={self.epoch:>2d},',
                    f'lr={self.current_lr:>.8f},',
                    f'tl={loss_sum / n_batches:>.4f},',
                    f'r@1/5/10={recalls[0]:.2f}/{recalls[1]:.2f}/{recalls[2]:.2f}',
                    '\n' if not_improved else ' *\n',
                ])
            else:
                recalls = None

            self.save_model(self.student_net, is_best=False, save_every_epoch=True)
            if self.opt.patience > 0 and not_improved > self.opt.patience:
                print('terminated because performance has not improve for', self.opt.patience, 'epochs')
                break

        print('best r@1/5/10={:.2f}/{:.2f}/{:.2f}'.format(self.best_recalls[0], self.best_recalls[1],
                                                          self.best_recalls[2]))
        return self.best_recalls

    def val(self, model):
        recalls, _ = self.get_recall(model)
        for i, n in enumerate([1, 5, 10]):
            wandb.log({'{}/{}_r@{}'.format(model.id, self.opt.split, n): recalls[i]}, step=self.step)
            # self.writer.add_scalar('{}/{}_r@{}'.format(model.id, self.opt.split, n), recalls[i], self.epoch)

        return recalls

    def test(self):
        # recalls, _ = self.get_recall(self.model, save_embs=True)
        # print('best r@1/5/10={:.2f}/{:.2f}/{:.2f}'.format(recalls[0], recalls[1], recalls[2]))
        self.test4image(self.model)
        # return recalls
        return None

    def input_transform(self, opt=None):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((opt.height, opt.width), interpolation=InterpolationMode.BILINEAR),
        ])

    def test4image(self, model):
        model.eval()

        if self.opt.split == 'val':
            eval_dataloader = self.whole_val_data_loader
            eval_set = self.whole_val_set
        elif self.opt.split == 'test':
            eval_dataloader = self.whole_test_data_loader
            eval_set = self.whole_test_set
        # print(f"{self.opt.split} len:{len(eval_set)}")
        whole_mu = torch.zeros((len(eval_set), self.opt.output_dim), device=self.device)  # (N, D)
        whole_var = torch.zeros((len(eval_set), self.opt.sigma_dim), device=self.device)  # (N, D)
        mu_in = torch.zeros((1, self.opt.output_dim), device=self.device)
        whole_input = torch.zeros((len(eval_set), 1), device=self.device)
        gt = eval_set.get_positives()  # (N, n_pos)

        with torch.no_grad():
            inputimage_path = "pittsburgh/query/004/004828_pitch2_yaw11.jpg"
            input_transform = self.input_transform(self.opt)
            inputimage = input_transform(Image.open(inputimage_path))
            inputimage_device = inputimage.unsqueeze(0)
            inputimage_device = inputimage_device.to(self.device)
            mu_inputimage, var_inputimage = model(inputimage_device)
            mu_in[0, :] = mu_inputimage

            del mu_inputimage, var_inputimage
            for iteration, (input, indices) in enumerate(tqdm(eval_dataloader), 1):
                input = input.to(self.device)
                mu, var = model(input)  # (B, D)
                # print(input)                    #(128,3,224,224)
                # var = torch.exp(var)

                whole_mu[indices, :] = mu
                whole_var[indices, :] = var
                del input, mu, var
        n_values = [3]

        whole_var = torch.exp(whole_var)
        whole_mu = whole_mu.cpu().numpy()
        whole_var = whole_var.cpu().numpy()
        mu_in = mu_in.cpu().numpy()
        # print(mu_in.shape)
        mu_inquery = mu_in[:1].astype('float32')
        # print(mu_inquery.shape)
        mu_q = whole_mu[eval_set.dbStruct.numDb:].astype('float32')
        # print(mu_q.shape)
        mu_db = whole_mu[:eval_set.dbStruct.numDb].astype('float32')
        sigma_q = whole_var[eval_set.dbStruct.numDb:].astype('float32')
        sigma_db = whole_var[:eval_set.dbStruct.numDb].astype('float32')
        faiss_index = faiss.IndexFlatL2(mu_q.shape[1])
        faiss_index.add(mu_db)
        dists, preds = faiss_index.search(mu_q, max(n_values))  # the results is sorted

        dists_input, preds_input = faiss_index.search(mu_inquery, max(n_values))
        print(preds_input[0, 0])
        print(dists_input[0, 0])
        pair_index = preds_input[0, 0]

        structFile = join(self.opt.structDir, 'pitts30k_test.mat')
        self.dbStruct = parse_dbStruct(structFile)
        image_pair_path = join('pittsburgh', 'database', self.dbStruct.dbImage[pair_index])
        print(image_pair_path)
        image_pair = Image.open(image_pair_path)
        image_pair.save('output_image.jpg')
        # img_dir=
        # path=join(img_dir, 'database', dbIm)

        return None

    def save_model(self, model, is_best=False, save_every_epoch=False):
        if is_best:
            torch.save({
                'epoch': self.epoch,
                'step': self.step,
                'state_dict': model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
            }, os.path.join(self.opt.runsPath, 'ckpt_best.pth.tar'))

        if save_every_epoch:
            torch.save({
                'epoch': self.epoch,
                'step': self.step,
                'state_dict': model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
            }, os.path.join(self.opt.runsPath, 'ckpt_e_{}.pth.tar'.format(self.epoch)))

    def get_recall(self, model, save_embs=False):
        model.eval()

        if self.opt.split == 'val':
            eval_dataloader = self.whole_val_data_loader
            eval_set = self.whole_val_set
        elif self.opt.split == 'test':
            eval_dataloader = self.whole_test_data_loader
            eval_set = self.whole_test_set
        # print(f"{self.opt.split} len:{len(eval_set)}")

        whole_mu = torch.zeros((len(eval_set), self.opt.output_dim), device=self.device)  # (N, D)
        whole_var = torch.zeros((len(eval_set), self.opt.sigma_dim), device=self.device)  # (N, D)
        gt = eval_set.get_positives()  # (N, n_pos)
        start_time = time.time()
        with torch.no_grad():
            for iteration, (input, indices) in enumerate(tqdm(eval_dataloader), 1):
                # print(f"Batch {iteration}, Indices: {indices}")
                input = input.to(self.device)
                mu, var = model(input)  # (B, D)
                # summary(self.model, input_size=input.shape[1:])
                # print(input.shape)
                # var = torch.exp(var)
                whole_mu[indices, :] = mu
                whole_var[indices, :] = var
                del input, mu, var
        end_time = time.time()

        elapsed_time = end_time - start_time
        print("Elapsed Time:", elapsed_time)
        n_values = [1, 5, 10]

        whole_var = torch.exp(whole_var)
        whole_mu = whole_mu.cpu().numpy()
        whole_var = whole_var.cpu().numpy()
        mu_q = whole_mu[eval_set.dbStruct.numDb:].astype('float32')
        mu_db = whole_mu[:eval_set.dbStruct.numDb].astype('float32')
        sigma_q = whole_var[eval_set.dbStruct.numDb:].astype('float32')
        sigma_db = whole_var[:eval_set.dbStruct.numDb].astype('float32')
        faiss_index = faiss.IndexFlatL2(mu_q.shape[1])
        faiss_index.add(mu_db)
        dists, preds = faiss_index.search(mu_q, max(n_values))  # the results is sorted

        # cull queries without any ground truth positives in the database
        val_inds = [True if len(gt[ind]) != 0 else False for ind in range(len(gt))]
        val_inds = np.array(val_inds)
        mu_q = mu_q[val_inds]
        sigma_q = sigma_q[val_inds]
        preds = preds[val_inds]
        dists = dists[val_inds]
        gt = gt[val_inds]

        recall_at_k = cal_recall(preds, gt, n_values)

        if save_embs:
            with open(join(self.opt.runsPath, '{}_db_embeddings_{}.pickle'.format(self.opt.split,
                                                                                  self.opt.resume.split('.')[-3].split(
                                                                                          '_')[-1])), 'wb') as handle:
                pickle.dump(mu_q, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(mu_db, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(sigma_q, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(sigma_db, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(preds, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(dists, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(gt, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(whole_mu, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(whole_var, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('embeddings saved for post processing')

        return recall_at_k, None


def parse_dbStruct(path):
    mat = loadmat(path)
    matStruct = mat['dbStruct'].item()

    dataset = 'nuscenes'

    whichSet = matStruct[0].item()

    # .mat file is generated by python, Kaiwen replaces the use of cell (in Matlab) with char (in Python)
    dbImage = [f[0].item() for f in matStruct[1]]
    # dbImage = matStruct[1]
    utmDb = matStruct[2].T
    # utmDb = matStruct[2]

    # .mat file is generated by python, I replace the use of cell (in Matlab) with char (in Python)
    qImage = [f[0].item() for f in matStruct[3]]
    # qImage = matStruct[3]
    utmQ = matStruct[4].T
    # utmQ = matStruct[4]

    numDb = matStruct[5].item()
    numQ = matStruct[6].item()

    posDistThr = matStruct[7].item()
    posDistSqThr = matStruct[8].item()
    nonTrivPosDistSqThr = matStruct[9].item()

    return dbStruct(whichSet, dataset, dbImage, utmDb, qImage, utmQ, numDb, numQ, posDistThr, posDistSqThr,
                    nonTrivPosDistSqThr)


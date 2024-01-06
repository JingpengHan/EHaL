import os
import time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable
from torchvision import transforms
from torch.optim import lr_scheduler
from MAP_calculate import *

def get_scheduler(optimizer, opt):
    #Return a learning rate scheduler
    if opt.lr_policy == 'linear':

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer,
                                        step_size=opt.lr_decay_iters,
                                        gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.2,
                                                   threshold=0.01,
                                                   patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=opt.n_epochs,
                                                   eta_min=0)
    else:
        return NotImplementedError(
            'learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

class ResidualBlock(nn.Module):
    #define residual blocks of the resnet
    def __init__(self, dim_in, dim_out, net_mode=None):

        if net_mode == 'p' or (net_mode is None):
            use_affine = True
        elif net_mode == 't':
            use_affine = False

        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in,
                      dim_out,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False), nn.InstanceNorm2d(dim_out, affine=use_affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out,
                      dim_out,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False), nn.InstanceNorm2d(dim_out, affine=use_affine))

    def forward(self, x):
        return x + self.main(x)

class classencryptor(nn.Module):
    #the architecture of class encryptor
    def __init__(self, bit, num_classes):
        super(classencryptor, self).__init__()

        self.feature = nn.Sequential(nn.Linear(num_classes, 4096),nn.ReLU(True), nn.Linear(4096, 512))
        self.hashing = nn.Sequential(nn.Linear(512, bit), nn.Tanh())
        self.classifier = nn.Sequential(nn.Linear(512, num_classes),nn.Sigmoid())

    def forward(self, label):
        f = self.feature(label)
        h = self.hashing(f)
        c = self.classifier(f)
        return f, h, c

class Generator(nn.Module):
    #the architecture of encrypted patch generator
    def __init__(self):
        super(Generator, self).__init__()
        self.label_encoder = LabelEncoder()
        curr_dim = 64
        image_encoder = [
            nn.Conv2d(6, curr_dim, kernel_size=7, stride=1, padding=3, bias=True),
            nn.InstanceNorm2d(curr_dim),
            nn.ReLU(inplace=True)
        ]
        # input images
        for i in range(2):
            image_encoder += [
                nn.Conv2d(curr_dim,
                          curr_dim * 2,
                          kernel_size=4,
                          stride=2,
                          padding=1,
                          bias=True),
                nn.InstanceNorm2d(curr_dim * 2),
                nn.ReLU(inplace=True)
            ]
            curr_dim = curr_dim * 2
        # Bottleneck
        for i in range(3):
            image_encoder += [
                ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, net_mode='t')
            ]
        self.image_encoder = nn.Sequential(*image_encoder)
        decoder = []
        # Bottleneck
        for i in range(3):
            decoder += [
                ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, net_mode='t')
            ]
        # out images
        for i in range(2):
            decoder += [
                nn.ConvTranspose2d(curr_dim,
                                   curr_dim // 2,
                                   kernel_size=4,
                                   stride=2,
                                   padding=1,
                                   bias=False),
                nn.InstanceNorm2d(curr_dim // 2),
                nn.ReLU(inplace=True)
            ]
            curr_dim = curr_dim // 2
        self.residual = nn.Sequential(
            nn.Conv2d(curr_dim + 3,
                      3,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False), nn.Tanh())
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x, label_feature):
        mixed_feature = self.label_encoder(x, label_feature)
        encode = self.image_encoder(mixed_feature)
        decode = self.decoder(encode)
        decode_x = torch.cat([decode, x], dim=1)
        adv_x = self.residual(decode_x)
        return adv_x, mixed_feature


class Discriminator(nn.Module):
    #the architecture of encrypted patch discriminator
    def __init__(self, num_classes, image_size=224, conv_dim=64, repeat_num=5):
        super(Discriminator, self).__init__()

        layers = []
        layers.append(spectral_norm(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1)))
        layers.append(nn.LeakyReLU(0.01))
        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(spectral_norm(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1)))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2
        kernel_size = int(image_size / (2**repeat_num))
        self.main = nn.Sequential(*layers)
        self.fc = nn.Conv2d(curr_dim, num_classes + 1, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        h = self.main(x)
        out = self.fc(h)
        return out.squeeze()


class LabelEncoder(nn.Module):
    # encryption Label feature extraction and encoder
    def __init__(self, nf=128):
        super(LabelEncoder, self).__init__()

        self.nf = nf
        curr_dim = nf
        self.size = 14
        self.fc = nn.Sequential(nn.Linear(512, curr_dim * self.size * self.size), nn.ReLU(True))
        transform = []
        for i in range(4):
            transform += [
                nn.ConvTranspose2d(curr_dim,
                                   curr_dim // 2,
                                   kernel_size=4,
                                   stride=2,
                                   padding=1,
                                   bias=False),
                nn.InstanceNorm2d(curr_dim // 2, affine=False),
                nn.ReLU(inplace=True)
            ]
            curr_dim = curr_dim // 2
        transform += [
            nn.Conv2d(curr_dim,
                      3,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False)
        ]
        self.transform = nn.Sequential(*transform)

    def forward(self, image, label_feature):
        label_feature = self.fc(label_feature)
        label_feature = label_feature.view(label_feature.size(0), self.nf, self.size, self.size)
        label_feature = self.transform(label_feature)
        mixed_feature = torch.cat((label_feature, image), dim=1)
        return mixed_feature

class GANLoss(nn.Module):
    #define different GAN objectives
    def __init__(self, gan_mode, target_origianl_label=0.0, target_encrypted_label=1.0):
        super(GANLoss, self).__init__()
        self.register_buffer('origianl_label', torch.tensor(target_origianl_label))
        self.register_buffer('encrypted_label', torch.tensor(target_encrypted_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, label, target_is_origianl):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_origianl (bool) - - if the ground truth label is for origianl images or encrypted images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """
        if target_is_origianl:
            origianl_label = self.origianl_label.expand(label.size(0), 1)
            target_tensor = torch.cat([label, origianl_label], dim=-1)
        else:
            encrypted_label = self.encrypted_label.expand(label.size(0), 1)
            target_tensor = torch.cat([label, encrypted_label], dim=-1)
        return target_tensor

    def __call__(self, prediction, label, target_is_origianl):
        #calculate loss of the given discriminator's output and grount truth labels.
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(label, target_is_origianl)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_origianl:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

def spectral_norm(module):
    SpectralNorm.apply(module)
    return module

class SpectralNorm(object):
    def __init__(self):
        self.name = "weight"
        self.power_iterations = 1

    def compute_weight(self, module):
        u = getattr(module, self.name + "_u")
        v = getattr(module, self.name + "_v")
        w = getattr(module, self.name + "_bar")
        height = w.data.shape[0]

        for _ in range(self.power_iterations):
            v.data = l2normalize(
                torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        return w / sigma.expand_as(w)

    @staticmethod
    def apply(module):
        name = "weight"
        fn = SpectralNorm()
        try:
            u = getattr(module, name + "_u")
            v = getattr(module, name + "_v")
            w = getattr(module, name + "_bar")

        except AttributeError:
            w = getattr(module, name)
            height = w.data.shape[0]
            width = w.view(height, -1).data.shape[1]
            u = Parameter(w.data.new(height).normal_(0, 1),
                          requires_grad=False)
            v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
            w_bar = Parameter(w.data)
            module.register_parameter(name + "_u", u)
            module.register_parameter(name + "_v", v)
            module.register_parameter(name + "_bar", w_bar)

        # remove w from parameter list
        del module._parameters[name]
        setattr(module, name, fn.compute_weight(module))
        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)
        return fn

    def remove(self, module):
        weight = self.compute_weight(module)
        delattr(module, self.name)
        del module._parameters[self.name + '_u']
        del module._parameters[self.name + '_v']
        del module._parameters[self.name + '_bar']
        module.register_parameter(self.name, Parameter(weight.data))

    def __call__(self, module, inputs):
        setattr(module, self.name, self.compute_weight(module))

def remove_spectral_norm(module):
    name = 'weight'
    for k, hook in module._forward_pre_hooks.items():

        if isinstance(hook, SpectralNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError("spectral_norm of '{}' not found in {}".format(name, module))

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

class PatchEncryptionModel(nn.Module):

    def __init__(self, args):
        super(PatchEncryptionModel, self).__init__()
        self.bit = args.bit
        self.num_classes = args.classes
        self.rec_w = args.rec_w
        self.ham_w = args.ham_w
        self.disc_w = args.disc_w
        self.batch_size = args.batch_size
        self.model_name = '{}_{}_{}_{}'.format(args.dataset, args.hash_method, args.backbone, args.bit)
        self.lr = args.lr
        self.args = args
        self._build_model()

    def _build_model(self):
        self.hashing_network = torch.load(os.path.join(self.args.save, self.model_name + '.pth')).cuda()
        self.hashing_network.eval()
        self.classencryptor_net = nn.DataParallel(classencryptor(self.bit, self.num_classes)).cuda()
        self.generator = nn.DataParallel(Generator()).cuda()
        self.discriminator = nn.DataParallel(Discriminator(num_classes=self.num_classes)).cuda()
        self.criterionGAN = GANLoss('lsgan').cuda()

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def generate_hash_code(self, data_loader, num_data):
    #generate hash codes using the trained deep hashing network
        B = torch.zeros(num_data, self.bit)
        self.train_labels = torch.zeros(num_data, self.num_classes)

        for it, data in enumerate(data_loader, 0):
            data_input = data[0]
            data_input = Variable(data_input.cuda())
            output = self.hashing_network(data_input)
            batch_size_ = output.size(0)
            u_ind = np.linspace(it * self.batch_size,
                                np.min((num_data,
                                        (it + 1) * self.batch_size)) - 1,
                                batch_size_,
                                dtype=int)
            B[u_ind, :] = torch.sign(output.cpu().data)
            self.train_labels[u_ind, :] = data[1]
        return B

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            if self.args.lr_policy == 'plateau':
                scheduler.step(0)
            else:
                scheduler.step()
        self.lr = self.optimizers[0].param_groups[0]['lr']

    def gradient_penalty(self, y, x):
        #compute gradient penalty
        weight = torch.ones(y.size()).cuda()
        dydx = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=weight, retain_graph=True, create_graph=True, only_inputs=True)[0]
        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)


    def train_classencryptor(self, train_loader, encrypt_labels, num_train):
        # train the class encryptor
        optimizer_l = torch.optim.Adam(self.classencryptor_net.parameters(), lr=self.args.lr, betas=(0.5, 0.999))
        epochs = self.args.n_epochs * 2
        epochs = 1
        steps = 800
        batch_size = 32
        lr_steps = epochs * steps
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_l, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)
        criterion_l2 = torch.nn.MSELoss()
        # hash codes of training set
        B = self.generate_hash_code(train_loader, num_train)
        tB = B.numpy()
        np.savetxt(os.path.join('log', 'train_hash_code_{}.txt'.format(self.model_name)), tB, fmt="%d")
        B = B.cuda()

        for epoch in range(epochs):
            for i in range(steps):
                select_index = np.random.choice(range(encrypt_labels.size(0)), size=batch_size)
                batch_encrypt_label = encrypt_labels.index_select(0, torch.from_numpy(select_index)).cuda()
                optimizer_l.zero_grad()
                S = CalcSim(batch_encrypt_label.cpu(), self.train_labels)
                _, encrypt_hash_c, label_pred = self.classencryptor_net(batch_encrypt_label)
                zeta_x = encrypt_hash_c.mm(Variable(B).t()) / 2
                sim_loss = (Variable(S.cuda()) * zeta_x - log_trick(zeta_x)).sum() / (num_train * batch_size)
                sim_loss = -sim_loss
                qua_loss = (torch.sign(encrypt_hash_c) - encrypt_hash_c).pow(2).sum() / (1e4 * batch_size)
                cla_loss = criterion_l2(label_pred, batch_encrypt_label)
                C_loss = sim_loss + cla_loss + qua_loss
                C_loss.backward()
                optimizer_l.step()
                if i % self.args.patches_freq == 0:
                    print('Training_epoch: {:2d}, step: {:3d}, lr: {:.5f}, C_loss:{:.5f}, qua_loss:{:.5f}, sim_loss: {:.5f}, cla_loss: {:.7f}'
                        .format(epoch, i, scheduler.get_last_lr()[0], C_loss, qua_loss, sim_loss,  cla_loss))
                scheduler.step()
        self.save_classencryptor()

    def save_classencryptor(self):
        torch.save(self.classencryptor_net.module.state_dict(),
            os.path.join(self.args.save, 'classencryptor_{}.pth'.format(self.model_name)))

    def test_classencryptor(self, encrypt_labels, database_loader, database_labels, num_database, num_test):
        encrypted_labels = np.zeros([num_test, self.num_classes])
        qB = np.zeros([num_test, self.bit])

        for i in range(num_test):
            select_index = np.random.choice(range(encrypt_labels.size(0)), size=1)
            batch_encrypt_label = encrypt_labels.index_select(0, torch.from_numpy(select_index))
            encrypted_labels[i, :] = batch_encrypt_label.numpy()[0]
            _, encrypt_hash_l, __ = self.classencryptor_net(batch_encrypt_label.cuda().float())
            qB[i, :] = torch.sign(encrypt_hash_l.cpu().data).numpy()[0]
        database_code_path = os.path.join('log', 'database_hash_code_{}.txt'.format(self.model_name))

        if os.path.exists(database_code_path):
            dB = np.loadtxt(database_code_path, dtype=np.float)

        else:
            dB = self.generate_hash_code(database_loader, num_database)
            dB = dB.numpy()
            np.savetxt(os.path.join('test_codes', 'database_hash_code_{}.txt'.format(self.model_name)), dB, fmt="%d")
        e_emap = CalcMap(qB, dB, encrypted_labels, database_labels.numpy())
        print('classencryptor_testMAP: %3.5f' % (e_emap))

    def train_generator(self, train_loader, encrypt_labels, train_labels, database_loader, database_labels, num_database, num_train, num_test):
        # Optimizers
        optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=self.args.lr, betas=(0.5, 0.999))
        optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.args.lr, betas=(0.5, 0.999))
        self.optimizers = [optimizer_g, optimizer_d]
        criterion_l2 = torch.nn.MSELoss()
        self.schedulers = [get_scheduler(opt, self.args) for opt in self.optimizers]

        # train and test class encryptor
        if os.path.exists(os.path.join(self.args.save, 'classencryptor_{}.pth'.format(self.model_name))):
            self.load_classencryptor()
        else:
            self.train_classencryptor(train_loader, encrypt_labels, num_train)
        self.classencryptor_net.eval()
        self.test_classencryptor(encrypt_labels, database_loader, database_labels, num_database, num_test)

        total_epochs = self.args.n_epochs + self.args.n_epochs_decay + 1
        for epoch in range(self.args.epoch_count, total_epochs):
            print('\nTraining epoch: {}, learning rate: {:.7f}'.format(epoch, self.lr))
            for i, data in enumerate(train_loader):
                origianl_input, batch_label, batch_ind = data
                origianl_input = set_input_images(origianl_input)
                batch_label = batch_label.cuda()
                select_index = np.random.choice(range(encrypt_labels.size(0)), size=batch_label.size(0))
                batch_encrypt_label = encrypt_labels.index_select(0, torch.from_numpy(select_index)).cuda()
                feature, encrypt_hash_l, _ = self.classencryptor_net(batch_encrypt_label)
                encrypt_hash_l = torch.sign(encrypt_hash_l.detach())
                encrypted_g, _ = self.generator(origianl_input, feature.detach())

                # update D
                if i % 3 == 0:
                    self.set_requires_grad(self.discriminator, True)
                    optimizer_d.zero_grad()
                    origianl_d = self.discriminator(origianl_input)
                    # stop backprop to the generator by detaching
                    encrypted_d = self.discriminator(encrypted_g.detach())
                    origianl_d_loss = self.criterionGAN(origianl_d, batch_label, True)
                    encrypted_d_loss = self.criterionGAN(encrypted_d, batch_encrypt_label, False)
                    d_loss = (origianl_d_loss + encrypted_d_loss) / 2
                    d_loss.backward()
                    optimizer_d.step()

                # update G
                self.set_requires_grad(self.discriminator, False)
                optimizer_g.zero_grad()
                encrypted_g_d = self.discriminator(encrypted_g)
                disc_loss = self.criterionGAN(encrypted_g_d, batch_encrypt_label, True)
                rec_loss = criterion_l2(encrypted_g, origianl_input)
                encrypt_hashing_g = self.hashing_network((encrypted_g + 1) / 2)
                hamming_loss = encrypt_hashing_g * encrypt_hash_l
                hamming_loss = torch.mean(hamming_loss)
                hamming_loss = (-hamming_loss + 1)

                # backpropagation
                g_loss = self.rec_w * rec_loss + self.ham_w *hamming_loss + self.disc_w*disc_loss
                g_loss.backward()
                optimizer_g.step()

                if i % self.args.patches_freq == 0:
                    self.patches(encrypted_g, 'train_patches/{}/{}/'.format(self.args.patches, self.model_name), str(epoch) + '_' + str(i) + '_encrypted')
                    self.patches(origianl_input, 'train_patches/{}/{}/'.format(self.args.patches, self.model_name), str(epoch) + '_' + str(i) + '_original')

                if i % self.args.print_freq == 0:
                    print('step: {:3d} rec_loss: {:.7f} ham_loss: {:.3f} disc_loss: {:.3f} G_loss: {:.3f} D_loss: {:.3f}'.format(i, rec_loss, hamming_loss, disc_loss, g_loss, d_loss ))

            self.update_learning_rate()

        self.save_generator()

    def save_generator(self):
        torch.save(self.generator.module.state_dict(),
            os.path.join(self.args.save, 'generator_{}_{}_{}.pth'.format(self.model_name, self.rec_w, self.disc_w)))

    def patches(self, image, patches_dir, name):
        if not os.path.exists(patches_dir):
            os.makedirs(patches_dir)
        image = image.cpu().detach()[0]
        image = transforms.ToPILImage()(image)
        image.convert(mode='RGB').save(os.path.join(patches_dir, name + '.jpg'), quality=100)


    def test_generator(self, encrypt_labels, database_loader, test_loader, database_labels, test_labels, num_database, num_test):
        qB = np.zeros([num_test, self.bit])
        encrypted_labels = np.zeros([num_test, self.num_classes])
        perceptibility = 0
        self.classencryptor_net.eval()
        self.generator.eval()
        start = time.time()

        for it, data in enumerate(test_loader):
            data_input, _, data_ind = data
            select_index = np.random.choice(range(encrypt_labels.size(0)), size=data_ind.size(0))
            while (encrypt_labels[select_index] == _).all():
                select_index = np.random.choice(range(encrypt_labels.size(0)), size=data_ind.size(0))
            batch_encrypt_label = encrypt_labels.index_select(0, torch.from_numpy(select_index))
            encrypted_labels[data_ind.numpy(), :] = batch_encrypt_label.numpy()
            data_input = set_input_images(data_input)
            feature = self.classencryptor_net(batch_encrypt_label.cuda())[0]
            encrypt_encrypted, mix_image = self.generator(data_input, feature)
            encrypt_encrypted = (encrypt_encrypted + 1) / 2
            data_input = (data_input + 1) / 2 
            encrypt_hashing = self.hashing_network(encrypt_encrypted)
            qB[data_ind.numpy(), :] = torch.sign(encrypt_hashing.cpu().data).numpy()
            perceptibility += F.mse_loss(data_input, encrypt_encrypted).data * data_ind.size(0)
            self.patches(encrypt_encrypted, 'test_patches/{}/{}/'.format(self.args.patches, self.model_name), str(it)+'_encrypted')
            self.patches(data_input, 'test_patches/{}/{}/'.format(self.args.patches, self.model_name), str(it)+'_original')
        end = time.time()
        
        print('Running time: %s Seconds'%(end-start))
        np.savetxt(os.path.join('test_codes', 'encrypt_label_{}_{}_{}.txt'.format(self.args.dataset, self.args.backbone, self.bit)), encrypted_labels, fmt="%d")
        np.savetxt(os.path.join('test_codes', 'test_code_{}_{}_{}.txt'.format(self.args.dataset, self.args.backbone, self.bit)), qB, fmt="%d")
        database_code_path = os.path.join('test_codes', 'database_hash_code_{}.txt'.format(self.model_name))
        if os.path.exists(database_code_path):
            dB = np.loadtxt(database_code_path, dtype=np.float)
        else:
            dB = self.generate_hash_code(database_loader, num_database)
            dB = dB.numpy()
        print('test_perceptibility: {:.7f}'.format(torch.sqrt(perceptibility/num_test)))
        e_omap = CalcMap(qB, dB, test_labels, database_labels.numpy())
        print('encrypted-original_MAP: %3.5f' % (e_omap))
        e_emap = CalcMap(qB, dB, encrypted_labels, database_labels.numpy())
        print('encrypted-encryption_MAP: %3.5f' % (e_emap))
        top10e_emap = CalcTopMap(qB, dB, encrypted_labels, database_labels.numpy(), 10)
        print('Test_topkencrypted-encryptionMAP: %3.5f' % (top10e_emap))

    def load_classencryptor(self):
        self.classencryptor_net.module.load_state_dict(
            torch.load(os.path.join(self.args.save, 'classencryptor_{}.pth'.format(self.model_name))))

    def load_generator(self):
        self.generator.module.load_state_dict(
            torch.load(os.path.join(self.args.save, 'generator_{}_{}_{}.pth'.format(self.model_name, self.rec_w, self.disc_w))))

    def load_model(self):
        self.load_classencryptor()
        self.load_generator()

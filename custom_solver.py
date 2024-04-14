from model import Generator
from model import Discriminator
from generic_dataset import CELEBA_FORMAT_DATASET, RAFD_FORMAT_DATASET, get_loader

import time
import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image


class CustomSolver():
    """Custom solver for training and testing StarGAN on an arbitrary number of datasets."""

    def __init__(self, datasets, vis_dataset, dir, **kwargs):
        # Model configuration
        self.image_size = kwargs.get("image_size", 128)
        self.g_conv_dim = kwargs.get("g_conv_dim", 64)      # number of conv filters in the first layer of G
        self.d_conv_dim = kwargs.get("d_conv_dim", 64)      # number of conv filters in the first layer of D
        self.g_repeat_num = kwargs.get("g_repeat_num", 6)   # number of residual blocks in G
        self.d_repeat_num = kwargs.get("d_repeat_num", 6)   # number of strided conv layers in D
        self.lambda_cls = kwargs.get("lambda_cls", 1)       # weight for classification loss
        self.lambda_rec = kwargs.get("lambda_rec", 10)      # weight for reconstruction loss
        self.lambda_gp = kwargs.get("lambda_gp", 10)        # weight for gradient penalty loss

        # Training configuration
        self.batch_size = kwargs.get("batch_size", 16)                  # mini-batch size
        self.num_iters = kwargs.get("num_iters", 200000)                # number of total iterations for training D
        self.num_iters_decay = kwargs.get("num_iters_decay", 100000)    # number of total iterations for lr decay
        self.g_lr = kwargs.get("g_lr", 0.0001)                          # learning rate for G
        self.d_lr = kwargs.get("d_lr", 0.0001)                          # learning rate for D
        self.n_critic = kwargs.get("n_critic", 5)                       # number of D updates per each G update
        self.lr_update_step = kwargs.get("lr_update_step", 1000)        # frequency for lr decay

        # Datasets configuration
        self.datasets = datasets
        self.init_full_label_paddings()

        # Workdir configuration
        self.dir = Path(dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.dir / "visualization").mkdir(parents=True, exist_ok=True)
        print(f'Workdir set at {self.dir}')

        # Logging, visualizing, checkpointing
        self.log_step = kwargs.get("log_step", 10)                      # log frequency
        self.vis_dataset = vis_dataset                                  # fixed inputs used for visualization
        self.vis_step = kwargs.get("vis_step", 1000)                    # fixed-input visualization frequency
        self.vis_batch_size = kwargs.get("vis_batch_size", 6)           # mini-batch size for visualization
        self.model_save_step = kwargs.get("model_save_step", 10000)     # model checkpointing frequency

        # Init model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.build_model()

    def init_full_label_paddings(self):
        self.sum_label_size = sum(dataset.label_size() for dataset in self.datasets)
        self.mask_vector_size = len(self.datasets)
        print(f'sum_label_size: {self.sum_label_size}, mask_vector_size: {self.mask_vector_size}')

        self.zeros_left = [0 for _ in self.datasets]
        self.zeros_right = [0 for _ in self.datasets]
        for i in range(len(self.datasets)):
            self.zeros_left[i] = sum(dataset.label_size() for dataset in self.datasets[:i])
            self.zeros_right[i] = sum(dataset.label_size() for dataset in self.datasets[i+1:])
        print(f'Zero padding left: {self.zeros_left}')
        print(f'Zero padding right: {self.zeros_right}')


    def build_model(self):
        """Create a generator and a discriminator."""
        self.G = Generator(self.g_conv_dim, self.sum_label_size + self.mask_vector_size, self.g_repeat_num)
        self.D = Discriminator(self.image_size, self.d_conv_dim, self.sum_label_size, self.d_repeat_num)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [0.5, 0.999])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [0.5, 0.999])
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')

        self.G.to(self.device)
        self.D.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = self.dir / "checkpoints" / f'{resume_iters}-G.ckpt'
        D_path = self.dir / "checkpoints" / f'{resume_iters}-D.ckpt'
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def get_full_label(self, label, dl_idx):
        zero_padding_left = torch.zeros(label.size(0), self.zeros_left[dl_idx])
        zero_padding_right = torch.zeros(label.size(0), self.zeros_right[dl_idx])
        dataset_mask = torch.zeros(label.size(0), len(self.datasets))
        dataset_mask[:, dl_idx] = 1

        full_label = torch.cat([zero_padding_left, label, zero_padding_right, dataset_mask], dim=1)
        return full_label

    def classification_loss(self, logit, target, dataset_format):
        """Compute binary or softmax cross entropy loss."""
        if dataset_format == CELEBA_FORMAT_DATASET:
            return F.binary_cross_entropy_with_logits(logit, target, reduction='sum') / logit.size(0)
        elif dataset_format == RAFD_FORMAT_DATASET:
            return F.cross_entropy(logit, target)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def generate_all_targets(self):
        targets, empty_target = [], torch.zeros(self.sum_label_size + self.mask_vector_size)

        for i, dataset in enumerate(self.datasets):
            for j in range(dataset.label_size()):
                current_target = empty_target.clone()
                current_target[self.zeros_left[i] + j] = 1
                current_target[self.sum_label_size + i] = 1
                targets.append(current_target)

        return targets

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def train_multi(self, resume_iters=None):
        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if resume_iters:
            start_iters = resume_iters
            self.restore_model(resume_iters)

        # Initialize dataloaders
        dataloaders = [get_loader(dataset, self.batch_size, "train", num_workers=2) for dataset in self.datasets]
        iterators = [iter(dataloader) for dataloader in dataloaders]

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):
            for dl_idx, dataloader in enumerate(dataloaders):
                dataset_format = self.datasets[dl_idx].dataset_format()

                # Fetch real images and labels.
                data_iter = iterators[dl_idx]
                try:
                    x_real, label_org = next(data_iter)
                except: # iterator reached end of seq
                    iterators[dl_idx] = iter(dataloaders[dl_idx])
                    x_real, label_org = next(iterators[dl_idx])

                # Generate target domain labels randomly.
                rand_idx = torch.randperm(label_org.size(0))
                label_trg = label_org[rand_idx]

                # Construct full-label (all attrs + mask vector)
                c_org = self.get_full_label(label_org.clone(), dl_idx)
                c_trg = self.get_full_label(label_trg.clone(), dl_idx)

                x_real = x_real.to(self.device)             # Input images.
                c_org = c_org.to(self.device)               # Original domain labels.
                c_trg = c_trg.to(self.device)               # Target domain labels.
                label_org = label_org.to(self.device)       # Full original domain labels.
                label_trg = label_trg.to(self.device)       # Full target domain labels.

                # ================================================================================ #
                #                             Train the discriminator                              #
                # ================================================================================ #

                # Compute loss with real images.
                out_src, out_cls = self.D(x_real)
                lb, rb =  self.zeros_left[dl_idx], self.sum_label_size - self.zeros_right[dl_idx]
                out_cls = out_cls[:, lb : rb]  # retrieve only relevant labels
                d_loss_real = -torch.mean(out_src)
                d_loss_cls = self.classification_loss(out_cls, label_org, dataset_format)

                # Compute loss with fake images.
                x_fake = self.G(x_real, c_trg)
                out_src, _ = self.D(x_fake.detach())
                d_loss_fake = torch.mean(out_src)

                # Compute loss for gradient penalty.
                alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
                x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
                out_src, _ = self.D(x_hat)
                d_loss_gp = self.gradient_penalty(out_src, x_hat)

                # Backward and optimize.
                d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Logging.
                loss = {}
                loss['D/loss_real'] = d_loss_real.item()
                loss['D/loss_fake'] = d_loss_fake.item()
                loss['D/loss_cls'] = d_loss_cls.item()
                loss['D/loss_gp'] = d_loss_gp.item()

                # ================================================================================ #
                #                               Train the generator                                #
                # ================================================================================ #

                if (i+1) % self.n_critic == 0:
                    # Original-to-target domain.
                    x_fake = self.G(x_real, c_trg)
                    out_src, out_cls = self.D(x_fake)
                    out_cls = out_cls[:, lb : rb]  # retrieve only relevant labels
                    g_loss_fake = -torch.mean(out_src)
                    g_loss_cls = self.classification_loss(out_cls, label_trg, dataset_format)

                    # Target-to-original domain.
                    x_reconst = self.G(x_fake, c_org)
                    g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                    # Backward and optimize.
                    g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Logging.
                    loss['G/loss_fake'] = g_loss_fake.item()
                    loss['G/loss_rec'] = g_loss_rec.item()
                    loss['G/loss_cls'] = g_loss_cls.item()

                # =================================================================================== #
                #                  Logging, visualizing, checkpointing, lr decay                      #
                # =================================================================================== #

                # Print out training info. (per dataset)
                if (i+1) % self.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Elapsed [{}], Iteration [{}/{}], Dataset [{}]".format(et, i+1, self.num_iters, dl_idx+1)
                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

            # Fixed-image visualization.
            if (i+1) % self.vis_step == 0:
                visualization_dir = self.dir / "visualization" / f'{i+1}'
                visualization_dir.mkdir(exist_ok=True)

                vis_dataloader = get_loader(self.vis_dataset, self.vis_batch_size, "test", num_workers=2)
                with torch.no_grad():
                    for j, x_vis in enumerate(vis_dataloader):
                        x_vis = x_vis.to(self.device)
                        x_fake_list = [x_vis]
                        for target in self.generate_all_targets():
                            target = torch.stack([target for _ in range(x_vis.size(0))]).to(self.device)
                            x_fake_list.append(self.G(x_vis, target))
                        x_concat = torch.cat(x_fake_list, dim=3)

                        vis_path = visualization_dir / f'{j+1}.jpg'
                        save_image(self.denorm(x_concat.data.cpu()), vis_path, nrow=1, padding=0)

                    print('Saved real and fake images into {}...'.format(visualization_dir))

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = self.dir / "checkpoints" / f'{i+1}-G.ckpt'
                D_path = self.dir / "checkpoints" / f'{i+1}-D.ckpt'
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints at iteration {} into {}...'.format(i+1, self.dir / "checkpoints"))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {:.8f}, d_lr: {:.8f}.'.format(g_lr, d_lr))

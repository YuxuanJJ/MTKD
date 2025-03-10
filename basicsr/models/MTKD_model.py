import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
import time
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
import torch
from torch.nn import functional as F


@MODEL_REGISTRY.register()
class MTKD(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(MTKD, self).__init__(opt)

        # define teacher1 network
        self.net_g_tea1 = build_network(opt['tea1_network_g'])
        self.net_g_tea1 = self.model_to_device(self.net_g_tea1)
        self.print_network(self.net_g_tea1)

        # load teacher1 pretrained models
        load_path_tea1 = self.opt['tea1_path'].get('pretrain_network_g', None)
        if load_path_tea1 is not None:
            param_key = self.opt['tea1_path'].get('param_key_g', 'params')
            self.load_network(self.net_g_tea1, load_path_tea1, self.opt['tea1_path'].get('strict_load_g', True), param_key)
        else:
            raise ValueError(f'Please load teacher model')

        # define teacher2 network
        self.net_g_tea2 = build_network(opt['tea2_network_g'])
        self.net_g_tea2 = self.model_to_device(self.net_g_tea2)
        self.print_network(self.net_g_tea2)

        # load teacher2 pretrained models
        load_path_tea2 = self.opt['tea2_path'].get('pretrain_network_g', None)
        if load_path_tea2 is not None:
            param_key = self.opt['tea2_path'].get('param_key_g', 'params')
            self.load_network(self.net_g_tea2, load_path_tea2, self.opt['tea2_path'].get('strict_load_g', True),
                                param_key)
        else:
            raise ValueError(f'Please load teacher model')

        # define teacher3 network
        self.net_g_tea3 = build_network(opt['tea3_network_g'])
        self.net_g_tea3 = self.model_to_device(self.net_g_tea3)
        self.print_network(self.net_g_tea3)

        # load teacher pretrained models
        load_path_tea3 = self.opt['tea3_path'].get('pretrain_network_g', None)
        if load_path_tea3 is not None:
            param_key = self.opt['tea3_path'].get('param_key_g', 'params')
            self.load_network(self.net_g_tea3, load_path_tea3, self.opt['tea3_path'].get('strict_load_g', True),
                              param_key)
        else:
            raise ValueError(f'Please load teacher model')

        # define DCTSwin network
        self.net_g_DCTSwin = build_network(opt['DCTSwin_network_g'])
        self.net_g_DCTSwin = self.model_to_device(self.net_g_DCTSwin)
        self.print_network(self.net_g_DCTSwin)

        # load teacher pretrained models
        load_path_DCTSwin = self.opt['DCTSwin_path'].get('pretrain_network_g', None)
        if load_path_DCTSwin is not None:
            param_key = self.opt['DCTSwin_path'].get('param_key_g', 'params')
            self.load_network(self.net_g_DCTSwin, load_path_DCTSwin, self.opt['DCTSwin_path'].get('strict_load_g', True),
                              param_key)
        else:
            raise ValueError(f'Please load teacher model')

        self.net_g_tea1.eval()
        self.net_g_tea2.eval()
        self.net_g_tea3.eval()
        self.net_g_DCTSwin.eval()

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        rgb_mean = (0.4488, 0.4371, 0.4040)
        self.img_range = 255.
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('dis_opt'):
            self.distill_loss_fn = build_loss(train_opt['dis_opt']).to(self.device)
        else:
            self.distill_loss_fn = None

        if train_opt.get('stu_opt'):
            self.student_loss_fn = build_loss(train_opt['stu_opt']).to(self.device)
        else:
            self.student_loss_fn = None

        if self.distill_loss_fn is None or self.student_loss_fn is None:
            raise ValueError('Please check losses.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()

        self.output_tea1 = self.net_g_tea1(self.lq)
        self.output_tea2 = self.net_g_tea2(self.lq)
        self.output_tea3 = self.net_g_tea3(self.lq)

        self.output_DCTSwin = self.net_g_DCTSwin(torch.cat([self.output_tea1, self.output_tea2, self.output_tea3], dim=1))
        self.output_DCTSwin = self.output_DCTSwin / self.img_range + self.mean.type_as(self.lq)

        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # distill loss
        distill_loss = self.distill_loss_fn(self.output, self.output_DCTSwin)
        l_total += distill_loss
        loss_dict['distill_loss'] = distill_loss
        # student loss
        student_loss = self.student_loss_fn(self.output, self.gt)
        l_total += student_loss
        loss_dict['student_loss'] = student_loss
        loss_dict['l_total'] = l_total

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if self.opt['network_g']['type'] == 'SwinIR':
            window_size = self.opt['network_g']['window_size']
            scale = self.opt.get('scale', 1)
            mod_pad_h, mod_pad_w = 0, 0
            _, _, h, w = self.lq.size()
            if h % window_size != 0:
                mod_pad_h = window_size - h % window_size
            if w % window_size != 0:
                mod_pad_w = window_size - w % window_size
            img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.eval()
                with torch.no_grad():
                    self.output = self.net_g_ema(img)
            else:
                self.net_g.eval()
                with torch.no_grad():
                    self.output = self.net_g(img)
                self.net_g.train()

            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]
        else:
            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.eval()
                with torch.no_grad():
                    self.output = self.net_g_ema(self.lq)
            else:
                self.net_g.eval()
                with torch.no_grad():
                    self.output = self.net_g(self.lq)
                self.net_g.train()

    def test_selfensemble(self):
        # TODO: to be tested
        # 8 augmentations
        # modified from https://github.com/thstkdgus35/EDSR-PyTorch

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        # prepare augmented data
        lq_list = [self.lq]
        for tf in 'v', 'h', 't':
            lq_list.extend([_transform(t, tf) for t in lq_list])

        # inference
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
        else:
            self.net_g.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
            self.net_g.train()

        # merge results
        for i in range(len(out_list)):
            if i > 3:
                out_list[i] = _transform(out_list[i], 't')
            if i % 4 > 1:
                out_list[i] = _transform(out_list[i], 'h')
            if (i % 4) % 2 == 1:
                out_list[i] = _transform(out_list[i], 'v')
        output = torch.cat(out_list, dim=0)

        self.output = output.mean(dim=0, keepdim=True)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                # print(img_name)
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
                    # print(name, calculate_metric(metric_data, opt_))
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict


    def get_density(self):
        weights = [w for name, w in self.net_g.named_parameters() if "weight" in name]
        num_features = sum([w.numel() for w in weights])
        density = sum([torch.sum(w != 0).item() for w in weights]) / num_features
        return density

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

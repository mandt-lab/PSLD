python main/train_sde.py +dataset=cifar10/cifar10_psld \
                     dataset.diffusion.data.root=\'/home/pandeyk1/datasets/\' \
                     dataset.diffusion.data.name='cifar10' \
                     dataset.diffusion.data.norm=True \
                     dataset.diffusion.data.hflip=True \
                     dataset.diffusion.model.score_fn.in_ch=6 \
                     dataset.diffusion.model.score_fn.out_ch=6 \
                     dataset.diffusion.model.score_fn.nf=128 \
                     dataset.diffusion.model.score_fn.ch_mult=[2,2,2] \
                     dataset.diffusion.model.score_fn.num_res_blocks=8 \
                     dataset.diffusion.model.score_fn.attn_resolutions=[16] \
                     dataset.diffusion.model.score_fn.dropout=0.15 \
                     dataset.diffusion.model.score_fn.progressive_input='residual' \
                     dataset.diffusion.model.score_fn.fir=True \
                     dataset.diffusion.model.score_fn.embedding_type='fourier' \
                     dataset.diffusion.model.sde.beta_min=8.0 \
                     dataset.diffusion.model.sde.beta_max=8.0 \
                     dataset.diffusion.model.sde.decomp_mode='lower' \
                     dataset.diffusion.model.sde.nu=4.01 \
                     dataset.diffusion.model.sde.gamma=0.01 \
                     dataset.diffusion.model.sde.kappa=0.04 \
                     dataset.diffusion.training.seed=0 \
                     dataset.diffusion.training.chkpt_interval=50 \
                     dataset.diffusion.training.mode='hsm' \
                     dataset.diffusion.training.fp16=False \
                     dataset.diffusion.training.use_ema=True \
                     dataset.diffusion.training.batch_size=16 \
                     dataset.diffusion.training.epochs=2500 \
                     dataset.diffusion.training.accelerator='gpu' \
                     dataset.diffusion.training.devices=8 \
                     dataset.diffusion.training.results_dir=\'/home/pandeyk1/psld_results/sota/uncond/psld_hsm_gamma=0.01_nu=4.01_cifar10_continuous_sfn=ncsnpp_nres=8_chmult=222/\' \
                     dataset.diffusion.training.workers=1 \
                     dataset.diffusion.training.chkpt_prefix=\"hsm_gamma=0.01_nu=4.01_cifar10_continuous_sfn=ncsnpp_3rdFeb\"

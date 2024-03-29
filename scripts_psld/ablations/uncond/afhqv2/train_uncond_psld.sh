python main/train_sde.py +dataset=afhqv2/afhqv2128_psld \
                     dataset.diffusion.data.root=\'/home/pandeyk1/datasets/afhqv2/\' \
                     dataset.diffusion.data.name='afhqv2' \
                     dataset.diffusion.data.norm=True \
                     dataset.diffusion.data.hflip=True \
                     dataset.diffusion.model.pl_module='sde_wrapper' \
                     dataset.diffusion.model.score_fn.in_ch=6 \
                     dataset.diffusion.model.score_fn.out_ch=6 \
                     dataset.diffusion.model.score_fn.nf=128 \
                     dataset.diffusion.model.score_fn.ch_mult=[1,2,2,2,3] \
                     dataset.diffusion.model.score_fn.num_res_blocks=2 \
                     dataset.diffusion.model.score_fn.attn_resolutions=[16] \
                     dataset.diffusion.model.score_fn.dropout=0.2 \
                     dataset.diffusion.model.sde.beta_min=8.0 \
                     dataset.diffusion.model.sde.beta_max=8.0 \
                     dataset.diffusion.model.sde.decomp_mode='lower' \
                     dataset.diffusion.model.sde.nu=4.01 \
                     dataset.diffusion.model.sde.gamma=0.01 \
                     dataset.diffusion.model.sde.kappa=0.04 \
                     dataset.diffusion.model.sde.numerical_eps=1e-9 \
                     dataset.diffusion.training.loss.name='psld_score_loss' \
                     dataset.diffusion.training.seed=0 \
                     dataset.diffusion.training.chkpt_interval=50 \
                     dataset.diffusion.training.mode='hsm' \
                     dataset.diffusion.training.fp16=False \
                     dataset.diffusion.training.use_ema=True \
                     dataset.diffusion.training.batch_size=8 \
                     dataset.diffusion.training.epochs=2000 \
                     dataset.diffusion.training.accelerator='gpu' \
                     dataset.diffusion.training.devices=8 \
                     dataset.diffusion.training.results_dir=\'/home/pandeyk1/psld_results/ablations/uncond/afhq128/es3sde_hsm_gamma=0.01_nu=4.01_afhq128_continuous_sfn=ncsnpp/\' \
                     dataset.diffusion.training.workers=1 \
                     dataset.diffusion.training.chkpt_prefix=\"hsm_ablation_gamma=0.01_nu=4.01_afhq128_20thFeb23\"
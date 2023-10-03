ulimit -n 64000
python main/eval/inpaint.py +dataset=afhqv2/afhqv2128_psld \
                     dataset.diffusion.data.root=\'/home/pandeyk1/datasets/afhqv2/\' \
                     dataset.diffusion.data.name='afhqv2' \
                     +dataset.diffusion.data.mask_path=\'/home/pandeyk1/datasets\' \
                     dataset.diffusion.data.norm=True \
                     dataset.diffusion.data.hflip=True \
                     dataset.diffusion.model.score_fn.in_ch=6 \
                     dataset.diffusion.model.score_fn.out_ch=3 \
                     dataset.diffusion.model.score_fn.nf=160 \
                     dataset.diffusion.model.score_fn.ch_mult=[1,2,2,3,3] \
                     dataset.diffusion.model.score_fn.num_res_blocks=2 \
                     dataset.diffusion.model.score_fn.attn_resolutions=[8,16] \
                     dataset.diffusion.model.score_fn.dropout=0.2 \
                     dataset.diffusion.model.sde.beta_min=8.0 \
                     dataset.diffusion.model.sde.beta_max=8.0 \
                     dataset.diffusion.model.sde.nu=4.0 \
                     dataset.diffusion.model.sde.gamma=0 \
                     dataset.diffusion.model.sde.kappa=0.04 \
                     dataset.diffusion.model.sde.decomp_mode='lower' \
                     dataset.diffusion.evaluation.seed=0 \
                     dataset.diffusion.evaluation.denoise=True \
                     dataset.diffusion.evaluation.sample_prefix='gpu' \
                     dataset.diffusion.evaluation.path_prefix="1000" \
                     dataset.diffusion.evaluation.devices=8 \
                     dataset.diffusion.evaluation.batch_size=16 \
                     dataset.diffusion.evaluation.stride_type='quadratic' \
                     dataset.diffusion.evaluation.sample_from='target' \
                     dataset.diffusion.evaluation.workers=1 \
                     dataset.diffusion.evaluation.chkpt_path=\'/home/pandeyk1/psld_results/sota/uncond/afhq128/es3sde_hsm_gamma=0_nu=4.0_afhq128_continuous_sfn=ddpmpp/checkpoints/cached_chkpts/es3sde-hsm_sota_gamma=0_nu=4.0_afhq128_25May23_nf=160_chmult=12233-epoch=1999-loss=0.0011.ckpt\' \
                     dataset.diffusion.evaluation.sampler.name="ip_em_sde" \
                     dataset.diffusion.evaluation.n_samples=50000 \
                     dataset.diffusion.evaluation.n_discrete_steps=250 \
                     dataset.diffusion.evaluation.save_path=\'/home/pandeyk1/psld_results/sota/uncond/afhq128/es3sde_hsm_gamma=0_nu=4.0_afhq128_continuous_sfn=ddpmpp/inpaint_results_val/\'

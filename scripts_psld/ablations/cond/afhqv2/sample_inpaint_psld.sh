python main/eval/inpaint.py +dataset=afhqv2/afhqv2128_es3sde \
                     dataset.diffusion.data.root=\'/home/pandeyk1/datasets/afhqv2/\' \
                     dataset.diffusion.data.name='afhqv2' \
                     +dataset.diffusion.data.mask_path=\'/home/pandeyk1/datasets\' \
                     dataset.diffusion.data.norm=True \
                     dataset.diffusion.data.hflip=True \
                     dataset.diffusion.model.score_fn.in_ch=6 \
                     dataset.diffusion.model.score_fn.out_ch=6 \
                     dataset.diffusion.model.score_fn.nf=128 \
                     dataset.diffusion.model.score_fn.ch_mult=[1,2,2,2,3] \
                     dataset.diffusion.model.score_fn.num_res_blocks=2 \
                     dataset.diffusion.model.score_fn.attn_resolutions=[16] \
                     dataset.diffusion.model.score_fn.dropout=0.2 \
                     dataset.diffusion.model.sde.beta_min=8.0 \
                     dataset.diffusion.model.sde.beta_max=8.0 \
                     dataset.diffusion.model.sde.nu=4.01 \
                     dataset.diffusion.model.sde.gamma=0.01 \
                     dataset.diffusion.model.sde.kappa=0.04 \
                     dataset.diffusion.model.sde.decomp_mode='lower' \
                     dataset.diffusion.evaluation.seed=0 \
                     dataset.diffusion.evaluation.sample_prefix='gpu' \
                     dataset.diffusion.evaluation.path_prefix="1000" \
                     dataset.diffusion.evaluation.devices=8 \
                     dataset.diffusion.evaluation.batch_size=1 \
                     dataset.diffusion.evaluation.stride_type='uniform' \
                     dataset.diffusion.evaluation.sample_from='target' \
                     dataset.diffusion.evaluation.workers=1 \
                     dataset.diffusion.evaluation.chkpt_path=\'/home/pandeyk1/es3sde_results/ablations/uncond/afhq128/es3sde_hsm_gamma=0.01_nu=4.01_afhq128_continuous_sfn=ncsnpp/checkpoints/es3sde-hsm_ablation_gamma=0.01_nu=4.01_afhq128_20thFeb23-epoch=1749-loss=0.0033.ckpt\' \
                     dataset.diffusion.evaluation.sampler.name="ip_em_sde" \
                     dataset.diffusion.evaluation.n_samples=32 \
                     dataset.diffusion.evaluation.n_discrete_steps=1000 \
                     dataset.diffusion.evaluation.save_path=\'/home/pandeyk1/es3sde_results/sota/cond/inpaint/es3sde_hsm_gamma=0.01_nu=4.01_afhqv2_continuous_sfn=ncsnpp_nres=2/dummy_samples/test/wild/\'

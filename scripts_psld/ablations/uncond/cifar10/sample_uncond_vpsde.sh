python main/eval/sample.py +dataset=cifar10/cifar10_vpsde \
                     dataset.diffusion.data.root=\'/home/pandeyk1/datasets/\' \
                     dataset.diffusion.data.name='cifar10' \
                     dataset.diffusion.data.norm=True \
                     dataset.diffusion.data.hflip=True \
                     dataset.diffusion.model.score_fn.in_ch=3 \
                     dataset.diffusion.model.score_fn.out_ch=3 \
                     dataset.diffusion.model.score_fn.nf=128 \
                     dataset.diffusion.model.score_fn.ch_mult=[1,2,2,2] \
                     dataset.diffusion.model.score_fn.num_res_blocks=2 \
                     dataset.diffusion.model.score_fn.attn_resolutions=[16] \
                     dataset.diffusion.model.score_fn.dropout=0.1 \
                     dataset.diffusion.model.sde.beta_min=0.1 \
                     dataset.diffusion.model.sde.beta_max=20 \
                     dataset.diffusion.evaluation.seed=0 \
                     dataset.diffusion.evaluation.sample_prefix='gpu' \
                     dataset.diffusion.evaluation.devices=8 \
                     dataset.diffusion.evaluation.save_path=\'/home/pandeyk1/psld_results/ablations/uncond/cifar10/vpsde_cifar10_continuous_sfn=ncsnpp/speedvsquality/em_quadratic/\' \
                     dataset.diffusion.evaluation.batch_size=16 \
                     dataset.diffusion.evaluation.stride_type='quadratic' \
                     dataset.diffusion.evaluation.sample_from='target' \
                     dataset.diffusion.evaluation.workers=1 \
                     dataset.diffusion.evaluation.chkpt_path=\'/home/pandeyk1/psld_results/ablations/uncond/cifar10/vpsde_cifar10_continuous_sfn=ncsnpp/checkpoints/cached_chkpts/vpsde-dsm_ablation_cifar10_5thJan23-epoch=1999-loss=0.0259.ckpt\' \
                     dataset.diffusion.evaluation.sampler.name="em_sde" \
                     dataset.diffusion.evaluation.n_samples=10000 \
                     dataset.diffusion.evaluation.n_discrete_steps=1000 \
                     dataset.diffusion.evaluation.path_prefix="1000"

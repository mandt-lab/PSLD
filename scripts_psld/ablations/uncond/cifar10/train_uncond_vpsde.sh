# CIFAR-10
python main/train_sde.py +dataset=cifar10/cifar10_vpsde \
                     dataset.diffusion.data.root=\'/home/pandeyk1/datasets/\' \
                     dataset.diffusion.data.name='cifar10' \
                     dataset.diffusion.data.norm=True \
                     dataset.diffusion.data.hflip=True \
                     dataset.diffusion.model.score_fn.in_ch=3 \
                     dataset.diffusion.model.score_fn.out_ch=3 \
                     dataset.diffusion.model.score_fn.nf=128 \
                     dataset.diffusion.model.score_fn.ch_mult=[1,2,2,2] \
                     dataset.diffusion.model.score_fn.num_res_blocks=4 \
                     dataset.diffusion.model.score_fn.attn_resolutions=[16] \
                     dataset.diffusion.model.score_fn.dropout=0.1 \
                     dataset.diffusion.training.seed=0 \
                     dataset.diffusion.training.fp16=False \
                     dataset.diffusion.training.use_ema=True \
                     dataset.diffusion.training.batch_size=32 \
                     dataset.diffusion.training.epochs=2000 \
                     dataset.diffusion.training.accelerator='gpu' \
                     dataset.diffusion.training.devices=[0,1,2,3] \
                     dataset.diffusion.training.results_dir=\'/home/pandeyk1/psld_results/ablations/uncond/vpsde_cifar10_continuous_sfn=ncsnpp_testnccl/\' \
                     dataset.diffusion.training.workers=1 \
                     dataset.diffusion.training.chkpt_prefix="dsm_ablation_cifar10_5thJan23"
import streamlit as st

import os
import sys
import torch
from torch import nn
from diffusers.utils import load_image, export_to_video
from dataclasses import dataclass
from einops import rearrange

VGEN_BASEPATH = "/data/checkpoints/HiGen/"

@dataclass
class Model:
    clip_encoder: nn.Module
    diffusion: nn.Module
    autoencoder: nn.Module
    unet: nn.Module
    zero_y: torch.Tensor
    cfg: dict

def build_model(cfg_update):
    from tools.modules.config import cfg
    from utils.assign_cfg import assign_signle_cfg
    from utils.registry_class import MODEL, EMBEDDER, AUTO_ENCODER, DIFFUSION
    import utils.transforms as data
    import tools.modules.unet


    for k, v in cfg_update.items():
        if isinstance(v, dict) and k in cfg:
            cfg[k].update(v)
        else:
            cfg[k] = v

    cfg_update['vldm_cfg'] = './VGen/configs/higen_train.yaml'
    cfg = assign_signle_cfg(cfg, cfg_update, 'vldm_cfg')

    for k, v in cfg_update.items():
        if isinstance(v, dict) and k in cfg:
            cfg[k].update(v)
        else:
            cfg[k] = v

    # [Diffusion]
    diffusion = DIFFUSION.build(cfg.Diffusion)

    # [Model] embedder
    cfg.embedder['pretrained'] = os.path.join(VGEN_BASEPATH, "open_clip_pytorch_model.bin")
    clip_encoder = EMBEDDER.build(cfg.embedder)
    clip_encoder.model.cuda()
    _, _, zero_y = clip_encoder(text="")
    zero_y = zero_y.detach()

    # [Model] auotoencoder 
    cfg.auto_encoder['pretrained'] = os.path.join(VGEN_BASEPATH, "v2-1_512-ema-pruned.ckpt")
    autoencoder = AUTO_ENCODER.build(cfg.auto_encoder)
    autoencoder.eval() # freeze
    for param in autoencoder.parameters():
        param.requires_grad = False
    autoencoder.cuda()
    
    # [Model] UNet 
    unet = MODEL.build(cfg.UNet)
    state_dict = torch.load(cfg.test_model, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    status = unet.load_state_dict(state_dict, strict=True)
    unet.cuda()
    unet.eval()

    return Model(
        diffusion=diffusion,
        clip_encoder=clip_encoder,
        autoencoder=autoencoder,
        unet=unet,
        zero_y=zero_y,
        cfg=cfg
    )

    
def higen():
    # Source: https://www.modelscope.cn/iic/HiGen.git
    model_path = os.path.join(VGEN_BASEPATH, "cvpr2024.t2v.e003.non_ema_0725000.pth")

    sys.path.append(os.path.join(os.path.dirname(__file__), 'VGen'))
    from utils.config import Config
    from utils.video_op import save_t2vhigen_video_safe

    cfg_update = Config(load=True, cfg_path='./VGen/configs/higen_infer.yaml')
    cfg_update.cfg_dict['test_model'] = model_path
    cfg_update.cfg_dict['debug'] = True # So that only one GPU is used.

    cfg = cfg_update.cfg_dict

    if 'model' not in st.session_state:
        st.session_state.model = None

    if st.session_state.model is None:
        print('Loading HiGen model')
        st.info('Loading HiGen model')
        model = build_model(cfg)
        st.session_state.model = model
        cfg = model.cfg

        st.info("Model loaded")
        st.rerun()
    else:
        st.info("Model is ready for video generation.")
        cfg = st.session_state.model.cfg
        prompt = st.text_input("Enter your text prompt:")

        if st.button('Generate'):
            st.info("Generating video")
            captions = [prompt]
            with torch.no_grad():
                _, y_text, y_words = st.session_state.model.clip_encoder(text=captions)

                spat_noise = torch.randn([1, 4, 1, int(cfg.resolution[1]/cfg.scale), int(cfg.resolution[0]/cfg.scale)]).cuda()
                spat_prior = torch.zeros_like(spat_noise).squeeze(2)
                motion_cond = torch.tensor([0], dtype=torch.long, device=spat_noise.device)
                appearance_cond = torch.Tensor([[[1.0]]]).repeat(1, 1, max(cfg.frame_lens)).cuda()
                model_kwargs=[
                    {'y': y_words, 'spat_prior': spat_prior, 'motion_cond': motion_cond, "appearance_cond": appearance_cond},
                    {'y': st.session_state.model.zero_y, 'spat_prior': spat_prior, 'motion_cond': motion_cond, "appearance_cond": appearance_cond}]
                
                spat_data = st.session_state.model.diffusion.ddim_sample_loop(
                    noise=spat_noise,
                    model=st.session_state.model.unet.eval(),
                    model_kwargs=model_kwargs,
                    guide_scale=cfg.guide_scale,
                    ddim_timesteps=cfg.ddim_timesteps,
                    eta=0.0)
                
                print('step 1')
                
                spat_key_frames = st.session_state.model.autoencoder.decode(1. / cfg.scale_factor * spat_data.squeeze(2))
                spat_data = spat_data.squeeze(2)

                temp_noise = torch.randn([1, 4, cfg.max_frames, int(cfg.resolution[1]/cfg.scale), int(cfg.resolution[0]/cfg.scale)]).cuda()
                b, c, f, h, w= temp_noise.shape
                offset_noise = torch.randn(b, c, f, 1, 1).cuda()
                temp_noise = temp_noise + cfg.noise_strength * offset_noise
                
                motion_cond = torch.tensor([[cfg.motion_factor] * (cfg.max_frames-1)], dtype=torch.long, device=temp_noise.device)

                sim_list = torch.cat([torch.linspace(1.0-cfg.appearance_factor, 1.0, cfg.max_frames)[:-1], torch.linspace(1.0, 1.0-cfg.appearance_factor, cfg.max_frames)])
                # sim_list = (torch.cos(sim_list * 3.1415926 + 3.1415926) + 1) / 2 # consine
                appearance_cond = torch.stack([sim_list[i:i+cfg.max_frames] for i in range(len(sim_list)-cfg.max_frames, -1, -1)]).cuda()
                # appearance_cond = CLIPSim().load_vid_sim('/mnt/data-nas-workspace/qingzhiwu/code/video_generation/workspace/temp_dir/cvpr2024_1.vidldm_15_pub_midj_unclip_basemodel_img_text_e003_eval_725000_pikachu_turn_back_g12/sample_000001/cvpr2024_1.vidldm_15_pub_midj_unclip_basemodel_img_text_e003_eval_725000_pikachu_turn_back_g12_s01_diff_0.0_500_.mp4')
                model_kwargs=[
                    {'y': y_words, 'spat_prior': spat_data, 'motion_cond': motion_cond, 'appearance_cond': appearance_cond[None, :]},
                    {'y': st.session_state.model.zero_y, 'spat_prior': spat_data, 'motion_cond': motion_cond, 'appearance_cond': appearance_cond[None, :]}]
                video_data = st.session_state.model.diffusion.ddim_sample_loop(
                    noise=temp_noise,
                    model=st.session_state.model.unet.eval(),
                    model_kwargs=model_kwargs,
                    guide_scale=cfg.guide_scale,
                    ddim_timesteps=cfg.ddim_timesteps,
                    eta=0.0)
                
                print('step 2')
                
            video_data = 1. / cfg.scale_factor * video_data # [1, 4, 32, 46]
            video_data = rearrange(video_data, 'b c f h w -> (b f) c h w')
            chunk_size = min(cfg.decoder_bs, video_data.shape[0])
            video_data_list = torch.chunk(video_data, video_data.shape[0]//chunk_size, dim=0)
            decode_data = []
            for vd_data in video_data_list:
                gen_frames = st.session_state.model.autoencoder.decode(vd_data)
                decode_data.append(gen_frames)
            video_data = torch.cat(decode_data, dim=0)
            video_data = rearrange(video_data, '(b f) c h w -> b c f h w', b = cfg.batch_size)
            print('step 3')

            text_size = cfg.resolution[-1]
            fn = f"/tmp/higen_{prompt.replace(' ','_')}.mp4"
            save_t2vhigen_video_safe(fn, video_data.cpu(), captions, cfg.mean, cfg.std, text_size)
            with open(fn, 'rb') as f:
                video_bytes = f.read()
                st.markdown('Generated video')
                st.video(video_bytes, format='video/mp4', start_time=0)

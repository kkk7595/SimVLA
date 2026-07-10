import os

from omegaconf import DictConfig

def get_model(cfg: DictConfig, torch_dtype=None):

    from rlinf.models.embodiment.simvla.simvla_action_model import SimVLAConfig
    from rlinf.models.embodiment.simvla.simvla_action_model import SimVLAForRLActionPrediction

    simvla_config:SimVLAConfig = SimVLAConfig(cfg=cfg)
    model: SimVLAForRLActionPrediction = SimVLAForRLActionPrediction(
        config=simvla_config
    )

    # from transformers import AutoConfig, CONFIG_MAPPING, PretrainedConfig

    # if "simvla" not in CONFIG_MAPPING:
    #     CONFIG_MAPPING["simvla"] = SimVLAConfig
    #     AutoConfig.register("simvla", SimVLAConfig)

    # model:SimVLAForRLActionPrediction
    # ckpt_path = cfg.smolvlm_model_path

    # if "HuggingFaceTB" in cfg.smolvlm_model_path or not os.path.isdir(ckpt_path):
    #     print("\n\n\n\n")
    #     print(f"##### load mode weight from {ckpt_path}")
    #     cfg.smolvlm_model_path =  "HuggingFaceTB/SmolVLM-500M-Instruct" 
    #     simvla_config:SimVLAConfig = SimVLAConfig(cfg=cfg)
    #     model: SimVLAForRLActionPrediction = SimVLAForRLActionPrediction(
    #         config=simvla_config
    #     )
    # else:
    #     print("\n\n\n\n")
    #     print(f"##### 22222222222")
    #     print(f"##### load mode weight from {ckpt_path}")
    #     model: SimVLAForRLActionPrediction = SimVLAForRLActionPrediction.from_pretrained(ckpt_path)

    return model

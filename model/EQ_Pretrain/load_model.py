import os
import hydra
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.initialize import hydra_init
from fairseq import tasks

@hydra.main(config_path=os.path.join(".", "fairseq", "config"), config_name="config")
def main(cfg: FairseqConfig) -> float:
    
    # Load Model for EQ-Pretrain
    hydra_init(cfg)
    task = tasks.setup_task(cfg.task)
    model_w2v = task.build_model(cfg.model)
    print(model_w2v)

    # Load Pre-trained Weights
    # Checkpoint = torch.load(your_model_path)
    # model.load_state_dict(Checkpoint["model"], strict=True)

    # Please run and edit your scripts within this block

if __name__ == "__main__":
    main()

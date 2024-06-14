import os
import torch
from transformers import AutoConfig


# PATHS
ROOT = os.path.dirname(__file__)
GLOVE = os.path.join(ROOT, "data/glove/glove.6B.100d.txt")
POLITIFACT_TITLE = os.path.join(ROOT, "data/politifact/politifact_title_no_ignore.tsv")
POLITIFACT_CONTENT = os.path.join(ROOT, "data/politifact/politifact_content_no_ignore.tsv")
POLITIFACT_COMMENT = os.path.join(ROOT, "data/politifact/politifact_comment_no_ignore.tsv")

# ARGS
ENCODERS = ['glove', 'bert']

# DEVICE
if torch.cuda.is_available():
    device_label = "cuda"
# elif torch.backends.mps.is_available():  # Apple silicon
#     device_label = "mps"
else:
    device_label = "cpu"
device = torch.device(device_label)


def defend_config(args) -> {}:
    preset = "google-bert/bert-base-uncased"
    # preset = "roberta-base"
    config = AutoConfig.from_pretrained(preset)
    model_save_dir = "saved/"
    model_save_name = f"defend_{args.encoder}.pth"

    return {
        'epochs': 50,
        'batch_size': 20,
        'lr': 0.001,
        'model_save_dir': model_save_dir,
        'model_save_path': os.path.join(model_save_dir, model_save_name),
        'encoder': args.encoder,
        'encoder_trainable': False,
        'max_comments': -1,               # No limit if <= 0
        'max_content_sentences': -1,      # No limit if <= 0
        'bert_pad_token_id': config.pad_token_id,
        'bert_preset': preset,
        'bert_token_count': config.max_position_embeddings,
        'bert_hidden_dim': config.dim if hasattr(config, 'dim') else config.hidden_size,
    }

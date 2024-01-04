import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
import json
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from modeling import ViPE
from utils import dotdict


def parse_args():
    parser = argparse.ArgumentParser(description="Training ViPE: Visualize Pretty-much Everything")

    parser.add_argument(
        "--model_name", type=str, default='gpt2-medium', help="which gpt2 version to use? [gpt2-medium or gpt2']"
    )

    parser.add_argument(
        "--data_set_file", type=str, default='/graphics/scratch2/staff/hassan/genuis_chatgpt/lyric_canvas.csv',
        help='path to lyricCanvas'
    )

    parser.add_argument(
        "--check_path", type=str, default='/graphics/scratch2/staff/hassan/checkpoints/lyrics_to_prompts/',
        help="path to save the model"
    )

    parser.add_argument(
        "--batch_size", type=int, default=32
    )

    parser.add_argument(
        "--epochs", type=int, default=5
    )

    parser.add_argument(
        "--learning_rate", type=float, default=5e-5
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=1e3
    )
    parser.add_argument(
        "--context_length", type=int, default=7, help='number of previous lines (lyrics) as a prefix'
    )

    parser.add_argument(
        "--device", type=str, default='cuda', help='cuda or cpu?'
    )

    args = parser.parse_args()
    return args


def main():
    # print('job is running')
    args = parse_args()

    hparams = dotdict({})

    hparams.model_name = args.model_name
    hparams.context_length = args.context_length
    hparams.batch_size = args.batch_size
    hparams.learning_rate = args.learning_rate
    hparams.device = args.device
    hparams.warmup_steps = args.warmup_steps
    max_epochs = args.epochs

    check_path = args.check_path
    check_path = check_path + '{}_v1.0/'.format(args.model_name)
    hparams.data_dir = args.data_set_file

    model_name = '{}_context_ctx_{}_lr_{}'.format(args.model_name, args.context_length, args.learning_rate)

    # Specify the directory path and file name
    file_path = check_path + 'hparams_' + model_name
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Open the file for writing
    with open(file_path, 'w') as file:
        file.write(json.dumps(hparams))

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=check_path + "logs/", name=model_name)
    checkpoint_callback = ModelCheckpoint(dirpath=check_path, save_top_k=5, monitor="val_loss", save_weights_only=True,
                                          filename=model_name)
    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=3)
    model = ViPE(hparams)
    model.to(args.device)

    trainer = Trainer(accelerator='gpu', devices=9, callbacks=[checkpoint_callback, early_stop], logger=tb_logger,
                      max_epochs=max_epochs, strategy='ddp')
    # trainer = Trainer(accelerator='gpu', devices=1, callbacks=[checkpoint_callback, early_stop], logger=tb_logger,   max_epochs=max_epochs, limit_train_batches=1000, limit_val_batches=10)
    trainer.fit(model)

if __name__ == "__main__":
    main()

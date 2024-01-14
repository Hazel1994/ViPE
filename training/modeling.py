from pytorch_lightning import LightningModule
from utils import Dataset,ContextAwareDataCollator
from transformers import GPT2LMHeadModel, AdamW, GPT2Tokenizer,get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.optim import AdamW

class ViPE(LightningModule):

    def __init__(self, hparams):
        super(ViPE, self).__init__()

        self.params=hparams
        self.data_dir=hparams.data_dir
        self.model = GPT2LMHeadModel.from_pretrained(hparams.model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(hparams.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.context_length=hparams.context_length
        self.batch_size=hparams.batch_size
        self.learning_rate=hparams.learning_rate
        self.adam_epsilon = 1e-8
        self.warmup_steps= hparams.warmup_steps
        self.weight_decay=0

    def forward(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch['token_type_ids']

        labels = input_ids.clone()
        # dont predict the lyrics, the loss is computed for the prompts only (token_type_ids==1)
        labels[token_type_ids == 0] = -100

        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss
        self.log("loss", loss.item(), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self(batch).loss
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        #no_decay = ["bias", "LayerNorm.weight"]
        # optimizer_grouped_parameters = [
        #     {
        #         "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        #         "weight_decay": self.weight_decay,
        #     },
        #     {
        #         "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        #         "weight_decay": 0.0,
        #     },
        # ]
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, eps=self.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
                'reduce_on_plateau': False
            }
        }

        ####################
        # DATA RELATED HOOKS
        ####################

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        self.train_dataset = Dataset(self.data_dir,context_size=self.context_length, training=True)
        self.valid_dataset = Dataset(self.data_dir,context_size=self.context_length,  training=False)
        self.data_collator = ContextAwareDataCollator(self.tokenizer)

    def train_dataloader(self):

        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                      shuffle=True, num_workers=4, collate_fn=self.data_collator, prefetch_factor=3)
        return train_dataloader

    def val_dataloader(self):
        valid_dataloader = DataLoader(self.valid_dataset, batch_size=self.batch_size,
                                      shuffle=False, num_workers=4, collate_fn=self.data_collator, prefetch_factor=3)
        return valid_dataloader

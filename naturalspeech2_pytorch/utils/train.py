from naturalspeech2_pytorch import NaturalSpeech2, Model, CustomDataset
from audiolm_pytorch import EncodecWrapper
from naturalspeech2_pytorch.utils.tokenizer import Tokenizer
from phonemizer import phonemize
from phonemizer.separator import Separator
import re

from naturalspeech2_pytorch import custom_collate_fn
from transformers import Trainer, TrainingArguments
from pathlib import Path


class PhonemizeText:
    def __init__(self):
        pass
    
    def phonemize(self, text, language='en-us', separator=' h# '):
        return phonemize_text(text, language, separator)

def phonemize_text(text, language='en-us', separator=' h# '):
    try:
        return  f"{separator}" + phonemize(
            text,
            language=language,
            backend='espeak',
            separator=Separator(word=separator, phone=' '),
            strip=True,
            preserve_punctuation=True
        ) + f"{separator}"
    except RuntimeError as e:
        print(f"Phonemization error: {str(e)}")
        return None

def clean_text(text):
    text = re.sub(r'[^\w\s\'\-]', '', text)  # Remove punctuation except apostrophes and hyphens
    text = re.sub(r'\s+', ' ', text)  # Collapse spaces
    return text.lower().strip()  # Normalize case and trim spaces


def compute_metrics(eval_pred):
    predictions, _ = eval_pred

    return {
        "mse_loss": predictions[0][0],
        "aux_loss": predictions[1][0],
        "duration_loss": predictions[2][0],
        "pitch_loss": predictions[3][0],
    }


def main():
    # Model and diffusion setup
    model = Model(dim=128, 
                  depth=6,
                  dim_prompt=512,
                  cond_drop_prob=0.1,
                  condition_on_prompt=True)

    # Initialize codec
    codec = EncodecWrapper()
    tokenizer = Tokenizer(phonemizer=PhonemizeText(), text_cleaner=clean_text)

    # Initialize NaturalSpeech2
    diffusion = NaturalSpeech2(
        model=model,
        codec=codec,
        speaker_embedding_dim=192,
        context_embedding_dim=768,
        timesteps=1000,
        tokenizer=tokenizer,
        objective='v',
        target_sample_hz=24000
    )

    # Dataset
    train_dataset_folder = "/workspace/datasets/VCTK_train"
    train_dataset = CustomDataset(dataset_folder=train_dataset_folder)


    val_dataset_folder = "/workspace/datasets/VCTK_val"
    val_dataset = CustomDataset(dataset_folder=val_dataset_folder)

    ds_config = str(Path(__file__).resolve().parent / "dp_config.json")


    training_args = TrainingArguments(
        output_dir="/workspace/newest_ckpt",
        per_device_train_batch_size=32,
        evaluation_strategy="steps",  # Evaluate at the end of each epoch
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        fp16=True,
        num_train_epochs=100,
        save_steps=500,
        eval_steps=500,
        logging_steps=100,
        learning_rate=float(3e-5),
        warmup_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        deepspeed=ds_config,
        lr_scheduler_type="cosine",
    )

    # Initialize Trainer
    trainer = Trainer(
        model=diffusion,
        data_collator=custom_collate_fn,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train
    trainer.train()


if __name__ == '__main__':
    main()

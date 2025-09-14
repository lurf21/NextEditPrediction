import os
import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("unsloth", "wandb")
    .apt_install("git")
)

app = modal.App("NEP_finetuning", image=image)

@app.function(
    image=image,
    gpu=f"H100:1",
    volumes={
        "/root/.cache/huggingface": modal.Volume.from_name(
            "huggingface-cache", create_if_missing=True
        ),
    },
    timeout=60 * 60 * 24,
    secrets=[modal.Secret.from_name("huggingface-secret"), modal.Secret.from_name("wandb-secret")],
)
def train():
    import os
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
    
    username = ""
    model_name = ""

    import wandb
    from datetime import datetime, timedelta, timezone
    tz = timezone(timedelta(hours=8))
    time = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    wandb.login(key=os.environ["WANDB_API_KEY"])
    wandb.init(
        project="NEP",
        name=f"{model_name.split('/')[1]}-{time}",
    )
    
    from unsloth import FastLanguageModel
    import torch
    max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        use_exact_model_name = True,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = False,
        load_in_8bit = False,
        full_finetuning = False,
    )

    print(model.config._name_or_path)

    tokenizer.add_tokens(["<|original_code|>", "<|edits_diff|>", "<|current_version|>", "<|next_version|>"], special_tokens=False)
    model.resize_token_embeddings(len(tokenizer))
    
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",
                        "lm_head", "embed_tokens",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    from datasets import load_dataset

    dataset = load_dataset(f"{username}/NextEditPrediction", split="train")
    
    # add eos_token in every example
    dataset = dataset.map(lambda x: {"text": x["text"] + tokenizer.eos_token}, remove_columns=["text"])
    # add input_ids in every example
    dataset = dataset.map(lambda x: tokenizer(x["text"], truncation = True, max_length = max_seq_length), remove_columns = ["text"])

    from transformers import TrainingArguments, DataCollatorForSeq2Seq
    from unsloth import UnslothTrainer, UnslothTrainingArguments, is_bfloat16_supported

    print(os.environ["UNSLOTH_IGNORED_TOKENIZER_NAMES"])

    trainer = UnslothTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
        dataset_num_proc = 16,
        packing = False, # Can make training 5x faster for short sequences.
        args = UnslothTrainingArguments(
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            num_train_epochs = 2, # Set this for 1 full training run.
            #max_steps = 10,
            learning_rate = 1e-4,
            embedding_learning_rate = 1e-5,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 10,
            optim = "paged_adamw_8bit", # Save more memory
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            report_to = "wandb", # Use this for WandB etc
            run_name = f"{model_name.split('/')[1]}-{time}",
        ),
    )

    from unsloth.chat_templates import train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|original_code|>\n",
        response_part = "<|next_version|>\n",
    )

    print(tokenizer.decode(trainer.train_dataset[0]["input_ids"]))
    space = tokenizer(" ", add_special_tokens = False).input_ids[0]
    print(tokenizer.decode([space if x == -100 else x for x in trainer.train_dataset[0]["labels"]]))

    print(trainer.train_dataset)

    trainer_stats = trainer.train()

    model.push_to_hub_merged(f"{username}/{model_name.split('/')[1]}-NEP", tokenizer, private=False, save_method="merged_16bit", token=os.environ["HF_TOKEN"])

if __name__ == "__main__":
    train()
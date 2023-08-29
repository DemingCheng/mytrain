import wandb
# 使用lora进行微调，速度更快，效果和直接微调原模型效果接近（通过实验验证）。
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_int8_training
import bitsandbytes as bnb

# Login with your authentication key
wandb.login()

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="myproject",
    dir="/data/intern/cmd/gpt2/mytrain"
    # track hyperparameters and run metadata
    # config={
    # "learning_rate": 0.02,
    # "architecture": "CNN",
    # "dataset": "CIFAR-100",
    # "epochs": 10,
    # }
)

from datasets import load_dataset
raw_data = load_dataset("text",data_files="./datasets/caregiver.txt")

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_size="left")
tokenizer.pad_token = tokenizer.eos_token
def tokenize_function(raw_data):
    tokenizer = AutoTokenizer.from_pretrained('gpt2', padding_size="left")
    tokenizer.pad_token = tokenizer.eos_token
    # return tokenizer(raw_data["text"], truncation=True)
    return tokenizer(raw_data["text"], padding=True, truncation=True)

tokenized_datasets = raw_data.map(tokenize_function, batched=True)
print(tokenized_datasets)

# prepair data collator
# from transformers import DataCollatorWithPadding
# collator = DataCollatorWithPadding(tokenizer=tokenizer)
from transformers import DataCollatorForLanguageModeling
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False)


from transformers import Trainer, TrainingArguments
training_args = TrainingArguments(
    output_dir='./results',          # 输出目录
    num_train_epochs=3,              # 训练轮数
    per_device_train_batch_size=5,   # 每个设备上的训练批次大小
    
#     warmup_steps (int，可选，默认为 0):
# 用于从 0 线性预热到 learning_rate 的步数。会覆盖 warmup_ratio 的任何效果。
    warmup_steps=100,           # 
    max_steps=20,             # 如果设置为正数，则要执行的总训练步骤数。会覆盖 num_train_epochs。如果使用有限的可迭代数据集，在耗尽所有数据时，训练可能会在达到设置的步数之前停止。
    learning_rate=3e-4          # 学习率123123
)

from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("gpt2", load_in_8bit=True)

# 设置lora配置参数
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=model,  # handled automatically by peft
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model = prepare_model_for_int8_training(model)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collator,
    train_dataset=tokenized_datasets["train"],
    tokenizer=tokenizer, 
)

trainer.train()

trainer.save_model(output_dir = "./model/mygpt2")


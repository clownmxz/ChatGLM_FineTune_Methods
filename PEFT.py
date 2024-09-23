# PEFT是一种基于LoRA的高效微调大模型库，相比于之前自己构建lora文件，这种方法可以直接调包

from ChatGLM_localized_deployment import Local_Model_For_PEFT
from pretrain_model.configuration_chatglm import ChatGLMConfig
from transformers import AutoTokenizer
from Accelerate import get_train_data, Seq2SeqDataset, collate_fn
from torch.utils.data import DataLoader
from accelerate import Accelerator
import torch
from tqdm import tqdm
from peft import get_peft_model, LoraConfig, TaskType, get_peft_model_state_dict

def print_trainable_parameter(model):
    """
    打印模型中可训练参数的数量，方便用于lora于非lora的对比
    """
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        # 获取参数的数量
        num_params = param.numel()
        # 如果参数数量为0，并且参数有ds_numel属性，则获取ds_numel属性值
        # ds_numel是LoRA模型中特有的属性，用于表示参数在LoRA中的数量
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        # 将参数数量累加到总参数数量中
        all_params += num_params
        # 如果参数需要梯度，则将参数数量累加到可训练参数数量中
        if param.requires_grad:
            trainable_params += num_params
    # 打印可训练参数数量、总参数数量和可训练参数占比
    print(f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params}")


if __name__ == '__main__':

    # 获取数据
    tokenizer = AutoTokenizer.from_pretrained("pretrain_model", trust_remote_code=True)
    prompt_text = "按照给定的格式抽取文本信息。\n文本："
    all_date = get_train_data("data/spo_0.json", tokenizer, 288, 256, prompt_text)
    train_dataset = Seq2SeqDataset(all_date)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=True, collate_fn=collate_fn,
                              num_workers=0)

    # 先进行一个用例测试
    model_path = "model_save/chatglm-6b-int4-localized.pt"  # 原预训练模型
    config = ChatGLMConfig()
    config.pre_seq_len = len(prompt_text)
    config.prefix_projection = False

    # 唯一需要主注意的地方，就是自定义模型的格式必须满足Hugging Face中的forward形式
    model = Local_Model_For_PEFT(model_path)
    peft_config = LoraConfig()
    model = get_peft_model(model, peft_config)
    model = model.half().cuda()

    # 查看一下只用lora需要多少参数
    print_trainable_parameter(model)

    # 开始训练，训练逻辑基本一致
    accelerator = Accelerator()
    device = accelerator.device

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, betas=(0.9, 0.99), eps=1e-5)
    lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2400, eta_min=2e-6, last_epoch=-1)

    model, optim, train_loader, lr_schedule = accelerator.prepare(model, optimizer, train_loader, lr_schedule)

    epochs = 20
    for epoch in range(epochs):
        pbar = tqdm(train_loader, total=len(train_loader))
        for batch in pbar:
            input_ids = batch["input_ids"].cuda()
            labels = batch["labels"].cuda()

            _, _, loss = model(input_ids, labels=labels)
            accelerator.backward(loss)

            optimizer.step()
            lr_schedule.step()
            optimizer.zero_grad()

            pbar.set_description(f"Epoch {epoch + 1}, Loss: {loss.item() :.4f}, lr: {lr_schedule.get_last_lr()[0]:.6f}")
        if (epoch + 1) % 4 == 0:
            # 这个是保存完整参数，不需要保存完整参数
            # torch.save(model.state_dict(), f"model_save/chatglm-6b-int4-accelerator-ft-{epoch + 1}.pt")
            peft_state_dict = get_peft_model_state_dict(model)
            torch.save(peft_state_dict, f"model_save/chatglm-6b-int4-peft-ft-{epoch + 1}.pt")







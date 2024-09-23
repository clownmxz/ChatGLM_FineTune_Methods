from ChatGLM_localized_deployment import Local_Model
from pretrain_model.configuration_chatglm import ChatGLMConfig
from pretrain_model.quantization import QuantizedLinear
from lora_file.model import add_lora
from lora_file.utils import get_lora_params, get_lora_state_dict
from transformers import AutoTokenizer
from Accelerate import get_train_data, Seq2SeqDataset, collate_fn
from torch.utils.data import DataLoader
from accelerate import Accelerator
import torch
from tqdm import tqdm

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

def generate(tokenizer, max_len, max_src_len, text, prompt_text, temperature=0.95, top_p=0.95):
    # 加载模型，是微调后的模型
    model_path = "model_save/chatglm-6b-int4-accelerator-ft-20.pt"
    config = ChatGLMConfig()
    config.pre_seq_len = len(prompt_text)
    config.prefix_projection = False
    model = Local_Model(model_path, config, strict=False)

    # 对于qkv层添加lora
    for key, _layer in model.named_modules():
        if "query_key_value" in key:
            add_lora(_layer)
    # 要冻结原有参数和添加层的参数，因为用于推断
    for name, param in model.named_parameters():
        param.requires_grad = False

    # 加载lora参数
    model.load_state_dict(torch.load("model_save/chatglm-6b-int4-lora-only-ft-20.pt"))
    model = model.half().cuda()


    model.eval()
    max_tgt_len = max_len - max_src_len

    with torch.no_grad():
        src_tokens = tokenizer.tokenize(text)
        prompt_tokens = tokenizer.tokenize(prompt_text)
        if len(src_tokens) > max_len - len(prompt_tokens):
            src_tokens = src_tokens[:max_len - len(prompt_tokens)]

        tokens = prompt_tokens + src_tokens + ["[gMASK]", "<sop>"]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        for _ in range(max_tgt_len):
            input_ids_tensor = torch.tensor([input_ids]).cuda()
            logits, _, _ = model.forward(input_ids_tensor)
            logits = logits[:, -3]
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = model.top_p_sampling(probs, top_p)
            input_ids = input_ids[:-2] + [next_token.item()] + input_ids[-2:]
            if next_token.item() == tokenizer.eos_token_id:
                break
        result = tokenizer.decode(input_ids)
        print(result)

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
    model = Local_Model(model_path, config, strict=False)
    model.glm_model.quantize(4)
    model = model.half().cuda()
    # 冻结原有模型的所有权重参数
    for name, param in model.named_parameters():
        param.requires_grad = False

    # 对于qkv层添加lora
    for key, _layer in model.named_modules():
        if "query_key_value" in key:
            add_lora(_layer)
    # 查看一下只用lora需要多少参数
    print_trainable_parameter(model)

    # 开始训练，训练逻辑基本一致
    accelerator = Accelerator()
    device = accelerator.device

    # 只记录lora的参数，用于后续的更新与保存
    lora_parameters = [{"params": list(get_lora_params(model))}]
    optimizer = torch.optim.AdamW(lora_parameters, lr=2e-5, betas=(0.9, 0.99), eps=1e-5)
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
            lora_state_dict = get_lora_state_dict(model)
            torch.save(lora_state_dict, f"model_save/chatglm-6b-int4-lora-only-ft-{epoch + 1}.pt")


    # 推断与载入，载入模型也只需要载入特定部分即可
    # 测试生成
    """
    text = "故障原因简要分析夜行灯，照明灯由同一开关（大灯组合开关TNS档位）控制。保险丝由一条共用主保险120A和照明灯独立保险5A、夜行灯独立保险" \
            "15A组成（线路图见附件二）。室内照明灯线路分布：手自一体开关、自动空调控制器、危险警报开关、音响控制单元、方向盘音响开关、组合仪表等" \
            "仪表台照明灯。夜行灯线路分布：左右前位置灯、左右后行车灯和牌照灯。可能原因：手自一体开关、自动空调控制器、危险警报开关、音响控制单元、" \
            "方向盘音响开关、组合仪表等故障，线路故障。维修方案处理方向盘音响按扭线束。"
    max_len = 288
    max_src_len = 256
    result = generate(tokenizer, max_len, max_src_len, text, prompt_text)
    """












# 使用Accelerate进行全量参数微调

import torch
import json
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from ChatGLM_localized_deployment import Local_Model
from pretrain_model.configuration_chatglm import ChatGLMConfig
from tqdm import tqdm


def get_train_data(filepath, tokenizer, max_len, max_src_len, prompt_text):
    max_tgt_len = max_len - max_src_len - 3
    all_data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            sample = json.loads(line.strip())
            # 注意tokenize和encode的区别
            src_tokens = tokenizer.tokenize(sample["text"])
            prompt_tokens = tokenizer.tokenize(prompt_text)
            # 注意切分长度
            if len(src_tokens) > max_src_len - len(prompt_tokens):
                src_tokens = src_tokens[:max_src_len - len(prompt_tokens)]
            tgt_tokens = tokenizer.tokenize("\n原因：" + sample["answer"])
            if len(tgt_tokens) > max_tgt_len:
                tgt_tokens = tgt_tokens[:max_tgt_len]

            # 符合ChatGLM的格式要求
            tokens = prompt_tokens + src_tokens + ["[gMASK]", "<sop>"] + tgt_tokens + ["<eop>"]

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            context_length = input_ids.index(tokenizer.bos_token_id)

            # 这里的labels就是 <sop> + tgt_tokens + <eop>，-100对应于交叉熵里的ignore_index
            labels = [-100] * context_length + input_ids[context_length:]

            # 填充符保持长度
            pad_len = max_len - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len
            assert len(input_ids) == max_len and len(labels) == max_len

            all_data.append({"text": sample["text"], "answer": sample["answer"], "input_ids": input_ids, "labels": labels})

    return all_data

def collate_fn(batch):
    input_ids_list, labels_list = [], []
    # 转换为tensor格式
    for instance in batch:
        input_ids_list.append(torch.tensor(instance["input_ids"], dtype=torch.long))
        labels_list.append(torch.tensor(instance["labels"], dtype=torch.long))
    # pad_sequence的作用是将多个序列按照指定的填充值填充到相同的长度，返回一个填充后的序列张量。
    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=3)
    labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)
    return {"input_ids": input_ids, "labels": labels}

class Seq2SeqDataset(Dataset):
    def __init__(self, all_data):
        self.all_data = all_data

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        return self.all_data[idx]

def generate(tokenizer, max_len, max_src_len, text, prompt_text, temperature=0.95, top_p=0.95):
    # 加载模型，是微调后的模型
    model_path = "model_save/chatglm-6b-int4-accelerator-ft-20.pt"
    config = ChatGLMConfig()
    config.pre_seq_len = len(prompt_text)
    config.prefix_projection = False
    model = Local_Model(model_path, config, strict=False)
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
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True, collate_fn=collate_fn, num_workers=0)

    # 加载模型
    model_path = "model_save/chatglm-6b-int4-localized.pt"
    config = ChatGLMConfig()
    config.pre_seq_len = len(prompt_text)
    config.prefix_projection = False
    model = Local_Model(model_path, config, strict=False)
    model = model.half().cuda()

    # 利用Accelerator进行全量参数微调
    accelerator = Accelerator()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, eps=1e-5)
    lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=2e-6)

    # accerator.prepare()函数用于将模型、优化器和数据加载器转换为可以在多个GPU上并行训练的格式。用法简单，这里和下面的损失。
    model, optim, train_loader, lr_schedule = accelerator.prepare(model, optimizer, train_loader, lr_schedule)

    epochs = 20
    for epoch in range(epochs):
        pbar = tqdm(train_loader, total=len(train_loader))
        for batch in pbar:
            input_ids = batch["input_ids"].cuda()
            labels = batch["labels"].cuda()

            _,_,loss = model(input_ids, labels=labels)
            accelerator.backward(loss)

            optimizer.step()
            lr_schedule.step()
            optimizer.zero_grad()

            pbar.set_description(f"Epoch {epoch + 1}, Loss: {loss.item() :.4f}, lr: {lr_schedule.get_last_lr()[0]:.6f}")
        if (epoch + 1) % 4 == 0:
            torch.save(model.state_dict(), f"model_save/chatglm-6b-int4-accelerator-ft-{epoch + 1}.pt")



    # 测试生成
    text = "故障原因简要分析夜行灯，照明灯由同一开关（大灯组合开关TNS档位）控制。保险丝由一条共用主保险120A和照明灯独立保险5A、夜行灯独立保险" \
           "15A组成（线路图见附件二）。室内照明灯线路分布：手自一体开关、自动空调控制器、危险警报开关、音响控制单元、方向盘音响开关、组合仪表等" \
           "仪表台照明灯。夜行灯线路分布：左右前位置灯、左右后行车灯和牌照灯。可能原因：手自一体开关、自动空调控制器、危险警报开关、音响控制单元、" \
           "方向盘音响开关、组合仪表等故障，线路故障。维修方案处理方向盘音响按扭线束。"
    max_len = 288
    max_src_len = 256
    result = generate(tokenizer, max_len, max_src_len, text, prompt_text)






import torch
import torch.nn as nn
from transformers import AutoTokenizer
from pretrain_model.configuration_chatglm import ChatGLMConfig
from pretrain_model.modeling_chatglm import ChatGLMForConditionalGeneration

class Local_Model(nn.Module):
    def __init__(self, model_path, config, strict=True):
        super(Local_Model, self).__init__()
        # 导入模型，包括模型框架，和保存的参数文件
        self.glm_model = ChatGLMForConditionalGeneration(config)
        self.glm_model.load_state_dict(torch.load(model_path), strict=strict)

        # 设置损失函数
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, position_ids=None, attention_mask=None, labels=None):
        # 前向传播
        logits, hidden_states = self.glm_model.forward(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask)
        # logits[batch_size, max_len, vocab_size]
        # 初始化loss为None
        loss = None
        # 如果labels不为None，这里就是微调的过程
        if labels is not None:
            # 将logits的最后一列去掉，并将结果进行连续化
            shift_logits = logits[:, :-1, :].contiguous()
            # 将labels的第二列开始的部分去掉，并将结果进行连续化
            shift_labels = labels[:, 1:].contiguous()

            # 将shift_logits的维度展平
            logits_1 = shift_logits.view(-1, shift_logits.size(-1))
            # 将shift_labels的维度展平
            labels_1 = shift_labels.view(-1)

            # 计算loss
            loss = self.loss_fn(logits_1, labels_1)

        return logits, hidden_states, loss

    def generate(self, prompt_text, continue_seq_length=128, tokenizer=None, temperature=0.95, top_p=0.95):

        # 做一个简单的问答系统或者其他问题
        if "？" in prompt_text:
            input_text = f"[Round 0]\n问：{prompt_text}\n答："
        else:
            input_text = prompt_text


        # 将输入文本进行编码
        input_ids = tokenizer.encode(input_text)
        # 不断生成文本
        for _ in range(continue_seq_length):
            # 转换为tensor格式
            input_ids_tensor = torch.tensor([input_ids]).to("cuda")
            # 预测生成
            with torch.no_grad():
                logits, _, _ = self.forward(input_ids=input_ids_tensor)
            # 获取最后一个token的logits
            logits = logits[:, -3]
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = self.top_p_sampling(probs, top_p)
            input_ids = input_ids[:-2] + [next_token.item()] + input_ids[-2:]
            if next_token.item() == 130005:
                print("break")
                break
        result = tokenizer.decode(input_ids)
        return result

    def top_p_sampling(self, probs, top_p):
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > top_p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))   # 归一化
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token


class Local_Model_For_PEFT(nn.Module):
    def __init__(self, model_path, grad_model_path=None):
        super().__init__()
        self.config = ChatGLMConfig()
        self.glm_model = ChatGLMForConditionalGeneration(self.config)
        self.glm_model.load_state_dict(torch.load(model_path))

        self.prepare_inputs_for_generation = self.glm_model.prepare_inputs_for_generation
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, position_ids=None, attention_mask=None,
                inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None,
                **kwargs):
        logits, hidden_states = self.glm_model.forward(input_ids=input_ids, **kwargs)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        # Flatten the tokens
        logits_1 = shift_logits.view(-1, shift_logits.size(-1))
        logits_2 = shift_labels.view(-1)

        loss = self.loss_fct(logits_1, logits_2)

        return logits, hidden_states, loss

    def generate(self, prompt_text, continue_seq_length=128, tokenizer=None, temperature=0.95, top_p=0.95):

        # 做一个简单的问答系统或者其他问题
        if "？" in prompt_text:
            input_text = f"[Round 0]\n问：{prompt_text}\n答："
        else:
            input_text = prompt_text

        # 将输入文本进行编码
        input_ids = tokenizer.encode(input_text)
        # 不断生成文本
        for _ in range(continue_seq_length):
            # 转换为tensor格式
            input_ids_tensor = torch.tensor([input_ids]).to("cuda")
            # 预测生成
            with torch.no_grad():
                logits, _, _ = self.forward(input_ids=input_ids_tensor)
            # 获取最后一个token的logits
            logits = logits[:, -3]
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = self.top_p_sampling(probs, top_p)
            input_ids = input_ids[:-2] + [next_token.item()] + input_ids[-2:]
            if next_token.item() == 130005:
                print("break")
                break
        result = tokenizer.decode(input_ids)
        return result

    def top_p_sampling(self, probs, top_p):
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > top_p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))  # 归一化
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token

if __name__ == '__main__':

    # 保存模型到本地
    """
    tokenizer = AutoTokenizer.from_pretrained("../pretrain_model/chatglm-6b-int4", trust_remote_code=True)
    model = AutoModel.from_pretrained("../pretrain_model/chatglm-6b-int4", trust_remote_code=True).half().cuda()

    response1, history = model.chat(tokenizer, "你好", history=[])
    response2, history = model.chat(tokenizer, "晚上睡不着怎么办？", history=history)
    print(response2)

    torch.save(model.state_dict(), "model_save/chatglm-6b-int4-localized.pt")
    """

    # 加载一下分词器
    tokenizer = AutoTokenizer.from_pretrained("pretrain_model", trust_remote_code=True, cache_dir="cache")
    # 加载一下模型
    model_path = "model_save/chatglm-6b-int4-localized.pt"

    config = ChatGLMConfig()
    model = Local_Model(model_path=model_path, config=config)
    model = model.half().cuda()

    # 进行问答
    prompt_text = "今天晚上吃什么菜好呢？"
    result = model.generate(prompt_text, continue_seq_length=256, tokenizer=tokenizer)
    print(result)








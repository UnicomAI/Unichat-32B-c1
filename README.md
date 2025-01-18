# Unichat-qwen2.5-32B-c1
---
[GitHub](https://github.com/UnicomAI/Unichat-32B-c1) | [ModelScope](https://www.modelscope.cn/UnicomAI/Unichat-qwen2.5-32B-c1.git)  |  [WiseModel](https://wisemodel.cn/models/UnicomLLM/Unichat-qwen2.5-32B-c1)

介绍
---
元景思维链模型，目前我们发布基于Qwen2.5-32B-Instruct实现的版本，后续将开源基于元景34B模型（UniChat 34B）的版本并公开我们的技术报告。

测评结果
---

| Model                | GSM8K | MATH500 | OlympidiaBench   | AIME2024 | AMC23 |
|-----------------------|---------------------|--------|-------|------------|------------|
| GPT-o1 mini   | 96.5                    | 93.7   | 78.8  | 66.7       | 92.5 |
|   GPT4o       | 90.4                    | 79.3   | 48.6  | 20.0       | 62.5 |
|  Deepseek V3  | 95.8                    | 90.2   | 50.1  | 40.0       | 80.0 |
| Qwen2.5-MATH-72B | 95.8                 | 85.9   | 49.0  | 30.0       | 70.0 |
|  Qwen-QwQ  | 95.6                       | 90.0   | 57.3  | 40.0       | 85.0 |
|  Unichat-qwen2.5-32B-c1   | 95.8                | 90.6   | 59.6  | 43.3       | 90.0 |

测评脚本使用[Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math)，推理长度设为12288

快速开始
---
这里提供代码片段来使用模型进行推理。
```python
from modelscope import AutoModelForCausalLM, AutoTokenizer

model_name = "UnicomAI/Unichat-32B-c1"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "9.8和9.11哪个数比较大？"
messages = [
    {"role": "system", "content": "请一步一步推理, 并把最终答案放在 \\boxed{{}} 里。"},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=8192
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)

```


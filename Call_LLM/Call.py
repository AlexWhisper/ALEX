from openai import OpenAI

def llama2_call(model,tokenizer,cur_prompt,start,device):
        # stime=time.time()
        # 将输入文本编码为模型输入
    input_ids = tokenizer.encode(cur_prompt, return_tensors="pt").to(device)
    output = model.generate(input_ids, max_length=input_ids.size()[1] + 100, num_return_sequences=1)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    rest = generated_text[start:]
    fa_index = rest.find('\n\nQuestion:')  # 找final_ans
    rf_index = rest.find('Generated answer:') #找到Generated answer

    if (fa_index > rf_index and rf_index != -1) or fa_index == -1:
        index = rf_index
    else:
        index = fa_index

    generate_q_a = rest[:index]

    return generate_q_a


def llama3_call(llamamodel,llamatokenizer,cur_prompt,start,device):
    llamatokenizer.pad_token_id = llamatokenizer.eos_token_id
    # stime=time.time()
    # 将输入文本编码为模型输入
    input_ids = llamatokenizer.encode(cur_prompt, return_tensors="pt").to(device)
    attention_mask = input_ids.ne(llamatokenizer.pad_token_id).int().to(llamamodel.device)
    output = llamamodel.generate(input_ids, max_length=input_ids.size()[1] + 100, num_return_sequences=1,attention_mask=attention_mask,pad_token_id=llamatokenizer.pad_token_id,eos_token_id=llamatokenizer.eos_token_id)
    generated_text = llamatokenizer.decode(output[0], skip_special_tokens=True)
    rest = generated_text[start:]
    fa_index = rest.find('\n\nQuestion:')  # 找final_ans
    rf_index = rest.find('Generated answer:') #找到Generated answer

    if (fa_index > rf_index and rf_index != -1) or fa_index == -1:
        index = rf_index
    else:
        index = fa_index

    generate_q_a = rest[:index]
    # print(generate_q_a)
    # etime = time.time()
    # ctime=etime-stime
    # print("\n调用callgpt函数耗时: ",ctime)
    return generate_q_a

from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=4, max=10))
def deepseek_call(model,tokenizer,cur_prompt,stop,device):

    # ans = openai.Completion.create(
    #     model="gpt-3.5-turbo-instruct",
    #     prompt=cur_prompt,
    #     temperature=0,
    #     stop=stop,
    #     max_tokens=200
    # )
    # returned = ans['choices'][0]['text']
    # return returned

    client = OpenAI(api_key="sk-96f5627c22254605af8e255339ef8737", base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        stop=stop,
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": cur_prompt},
        ],
        stream=False
    )
    returned=response.choices[0].message.content
    #print(returned)
    return returned



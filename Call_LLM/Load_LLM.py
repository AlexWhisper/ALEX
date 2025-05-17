from .Call import llama2_call,llama3_call,deepseek_call

method_map = {
    'Llama-2-7b-chat-hf': llama2_call,
    'llama3': llama3_call,
    'deepseek': deepseek_call
}

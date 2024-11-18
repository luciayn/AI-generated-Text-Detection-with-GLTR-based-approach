import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model.eval()
start_token = tokenizer(tokenizer.bos_token, return_tensors='pt').data['input_ids'][0]

def top_k_logits(logits, k):
    """
    Filters logits to only the top k choices
    from https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_gpt2.py
    """
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1]
    return torch.where(logits < min_values,
                       torch.ones_like(logits, dtype=logits.dtype) * -1e10,
                       logits)

def postprocess(token):
    with_space = False
    with_break = False
    if token.startswith('Ġ'):
        with_space = True
        token = token[1:]
        # print(token)
    elif token.startswith('â'):
        token = ' '
    elif token.startswith('Ċ'):
        token = ' '
        with_break = True

    token = '-' if token.startswith('â') else token
    token = '“' if token.startswith('ľ') else token
    token = '”' if token.startswith('Ŀ') else token
    token = "'" if token.startswith('Ļ') else token

    return token

def check_probabilities(in_text, topk = 100):
    # Process input
    token_ids = tokenizer(in_text, return_tensors='pt').data['input_ids'][0]
    token_ids = torch.concat([start_token, token_ids])

    # Forward through the model
    output = model(token_ids.to(device))
    all_logits = output.logits[:-1].detach().squeeze()
    # construct target and pred
    all_probs = torch.softmax(all_logits, dim=1)

    y = token_ids[1:]
    # Sort the predictions for each timestep
    sorted_preds = torch.argsort(all_probs, dim=1, descending=True).cpu()
    # [(pos, prob), ...]
    real_topk_pos = list(
        [int(np.where(sorted_preds[i] == y[i].item())[0][0])
          for i in range(y.shape[0])])
    real_topk_probs = all_probs[np.arange(
        0, y.shape[0], 1), y].data.cpu().numpy().tolist()
    real_topk_probs = list(map(lambda x: round(x, 5), real_topk_probs))

    real_topk = list(zip(real_topk_pos, real_topk_probs))
    # [str, str, ...]
    bpe_strings = tokenizer.convert_ids_to_tokens(token_ids[:])

    bpe_strings = [postprocess(s) for s in bpe_strings]

    topk_prob_values, topk_prob_inds = torch.topk(all_probs, k=topk, dim=1)

    pred_topk = [list(zip(tokenizer.convert_ids_to_tokens(topk_prob_inds[i]),
                          topk_prob_values[i].data.cpu().numpy().tolist()
                          )) for i in range(y.shape[0])]
    pred_topk = [[(postprocess(t[0]), t[1]) for t in pred] for pred in pred_topk]


    payload = {'bpe_strings': bpe_strings,
                'real_topk': real_topk,
                'pred_topk': pred_topk}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return payload


def color_text(text, payload):
    colored_text = ''
    l = len(text)
    count = {'Green (Top 10)': 0, 'Yellow (Top 100)': 0, 'Red (Top 1000)': 0, 'Purple (Others)': 0, 'Total': l}
    pred = 'Generated' # default generated

    for word in range(len(text)):
        color = ''
        if 0 <= payload['real_topk'][word][0] < 10:
            color = 'background-color: #CCFFCC;'  # Pastel green
            count['Green (Top 10)'] += 1
        elif 10 <= payload['real_topk'][word][0] < 100:
            color = 'background-color: #FFFF99;'  # Pastel yellow
            count['Yellow (Top 100)'] += 1
        elif 100 <= payload['real_topk'][word][0] < 1000:
            color = 'background-color: #FF9999;'  # Pastel red
            count['Red (Top 1000)'] += 1
        elif payload['real_topk'][word][0] >= 1000:
            color = 'background-color: #CC99FF;'  # Pastel purple
            count['Purple (Others)'] += 1
        
        colored_text += f'<span style="{color}">{text[word]}</span> '

    if count['Green (Top 10)'] > l*2/3:
      return colored_text, pred, count
    else:
      pred = 'Human' # human
      return colored_text, pred, count
    

def classify_text(in_text):
  # Tokenize input text
  tokens = tokenizer.tokenize(in_text)

  # Filter out special tokens and convert tokens to string
  text = [postprocess(token) for token in tokens if not tokenizer.special_tokens_map.get(token)]

  # Obtain probabilities
  payload = check_probabilities(in_text)

  # Color the text
  colored_text, pred, count = color_text(text, payload)

  return colored_text, pred, count


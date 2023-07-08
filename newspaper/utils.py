import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from tqdm import tqdm

batch_size = 16
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_newspaper(newspaper="20minutes"):
    if newspaper == "20minutes":
        #Load 20 minutes articles
        df_articles1 = pd.read_parquet('../newspaper_part1.parquet')
        df_articles2 = pd.read_parquet('../newspaper_part2.parquet')
        df_articles = pd.concat([df_articles1, df_articles2])
    elif newspaper == "liberation":
        df_articles = pd.read_parquet('../liberation.parquet')
    elif newspaper == "lepoint":
        df_articles = pd.read_parquet('../data/lepoint.parquet')
    else:
        print("The newspaper name must be 20minutes, liberation or lepoint")
        df_articles = pd.DataFrame()
    return df_articles 

def split_text_in_parts(txt):
    """
    Split articles in part of length 500 tokens max to be compatible with camembert model.
    """
    n = len(txt)
    prev_cursor = 0
    cursor = min(499, n-1)
    parts = []
    while prev_cursor < n-1:
        while '.' not in txt[cursor] and cursor > prev_cursor:
            cursor -= 1
        if cursor == prev_cursor:
            parts.append(txt[prev_cursor:min(prev_cursor+500, n)])
            prev_cursor = min(prev_cursor+500, n)
            cursor = min(prev_cursor+499, n-1)
        else:
            parts.append(txt[prev_cursor:cursor+1])
            prev_cursor = cursor+1
            cursor = prev_cursor+499
            if cursor >= n-1 and prev_cursor < n-1:
                parts.append(txt[prev_cursor:])
                break
    return parts

def extract_train_test_dataset(df_articles, level='body'):
    df_articles = df_articles.sort_values("article_date")
    df_articles = df_articles[(df_articles.body != '') & (df_articles.title != '')]
    dict_labels = {'planete': 0, 'sport': 1, 'economie': 2, 'arts-stars': 3, 'high-tech': 4, 'politique': 5, 'monde': 6, 'societe': 7, 'faits_divers': 8, 'sante': 9, 'justice': 10}
    p_train, p_test = df_articles[(df_articles.category_id == 'planete')].iloc[:10000],  df_articles[(df_articles.category_id == 'planete')].iloc[10000:11000]
    s_train, s_test = df_articles[(df_articles.category_id == 'sport')].iloc[:10000],  df_articles[(df_articles.category_id == 'sport')].iloc[10000:11000]
    e_train, e_test = df_articles[(df_articles.category_id == 'economie')].iloc[:10000],  df_articles[(df_articles.category_id == 'economie')].iloc[10000:11000]
    sc_train, sc_test = df_articles[(df_articles.category_id == 'arts-stars')].iloc[:10000],  df_articles[(df_articles.category_id == 'arts-stars')].iloc[10000:11000]
    h_train, h_test = df_articles[(df_articles.category_id == 'high-tech')].iloc[:10000],  df_articles[(df_articles.category_id == 'high-tech')].iloc[10000:11000]
    po_train, po_test = df_articles[(df_articles.category_id == 'politique')].iloc[:10000],  df_articles[(df_articles.category_id == 'politique')].iloc[10000:11000]
    m_train, m_test = df_articles[(df_articles.category_id == 'monde')].iloc[:10000],  df_articles[(df_articles.category_id == 'monde')].iloc[10000:11000]
    so_train, so_test = df_articles[(df_articles.category_id == 'societe')].iloc[:10000],  df_articles[(df_articles.category_id == 'societe')].iloc[10000:11000]
    fd_train, fd_test = df_articles[(df_articles.category_id == 'faits_divers')].iloc[:10000],  df_articles[(df_articles.category_id == 'faits_divers')].iloc[10000:11000]
    sa_train, sa_test = df_articles[(df_articles.category_id == 'sante')].iloc[:10000],  df_articles[(df_articles.category_id == 'sante')].iloc[10000:11000]
    j_train, j_test = df_articles[(df_articles.category_id == 'justice')].iloc[:10000],  df_articles[(df_articles.category_id == 'justice')].iloc[10000:11000]
    train_dataset = pd.concat([p_train, s_train, e_train, sc_train, h_train, po_train, m_train, so_train, fd_train, sa_train, j_train])[[level, 'category_id']]
    train_dataset['label'] = train_dataset.apply(lambda x: dict_labels[x['category_id']], axis=1)
    test_dataset = pd.concat([p_test, s_test, e_test, sc_test, h_test, po_test, m_test, so_test, fd_test, sa_test, j_test])[[level, 'category_id']]
    test_dataset['label'] = test_dataset.apply(lambda x: dict_labels[x['category_id']], axis=1)

    return train_dataset, test_dataset

def dataset_to_dataloader(dataset, tokenizer, level='body', details=False, labels=True):
    if level == 'body':
        MAX_LEN = 512
    else:
        MAX_LEN = 64
    text = dataset[level].to_list()
    if labels:
        labels = dataset['label'].to_list()
    else:
        labels = [0]*len(text)
    body_text = []
    body_labels = []
    body_id = []
    for i in range(len(text)):
        parts = split_text_in_parts(text[i].split())
        for part in parts:
            if part != []:
                body_text.append(' '.join(part))
                body_labels.append(labels[i])
                body_id.append(i)
        

    #user tokenizer to convert sentences into tokenizer
    input_ids  = tokenizer(body_text, max_length=MAX_LEN, padding='longest', truncation=True).input_ids

    # Create attention masks
    attention_masks = []
    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i!=1) for i in seq]  
        attention_masks.append(seq_mask)

    # transfrom to tensor format
    inputs_tensor = torch.tensor(input_ids)
    labels_tensor = torch.tensor(body_labels)
    masks_tensor = torch.tensor(attention_masks)

    # create dataloader
    data = TensorDataset(inputs_tensor, masks_tensor, labels_tensor)
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    if details:
        return dataloader, body_id, input_ids

    return dataloader

# function to compute accuracy

def compute_accuracy(test_dataloader, model):
    total_true = 0
    total_size = 0
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            t_data = batch[0].to(device)
            t_mask = batch[1].to(device)
            y = model(t_data,attention_mask=t_mask).logits
            result = torch.argmax(y, dim=-1).cpu().detach().numpy()
            nb_true, size = np.sum(result == np.array(batch[2])), len(result)
            total_true += nb_true
            total_size += size
        
    accuracy = total_true / total_size
    return accuracy

def train(model, train_dataloader, test_dataloader, optimizer, epochs=20, log_interval=50, level="body"):
    best_acc = 0
    losses = []
    for epoch in range(epochs):
        total_loss = 0
        for idx, batch in enumerate(train_dataloader):
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            optimizer.zero_grad()
            y_pred = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = y_pred[0]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if idx % log_interval == 0 and idx > 0:
                cur_loss = total_loss / log_interval
                print(
                    "| epoch {:3d} | {:5d}/{:5d} steps | "
                    "loss {:5.5f}".format(
                        epoch, idx, len(train_dataloader), cur_loss,
                    )
                )
                losses.append(cur_loss)
                total_loss = 0
        accuracy = compute_accuracy(test_dataloader, model)
        print("Test accuracy : {:1.3f}".format(accuracy))
        # Save model if better
        if accuracy > best_acc:
            torch.save(model.state_dict(), f'camembert_{level}.pt')
            best_acc = accuracy
    return model
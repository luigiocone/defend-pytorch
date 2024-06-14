import os
import sys

import evaluate
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_scheduler
from tqdm.auto import tqdm

from encoders.glove.Glove import Glove
from encoders.bert.BertTextDataset import BertTextDataset
from encoders.glove.GloveTextDataset import GloveTextDataset
from defend.SimpleDataCollator import SimpleDataCollator
from utils import metrics, politifact
from params import device, GLOVE
from defend.defend import Defend

import warnings
from bs4 import MarkupResemblesLocatorWarning

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])


def train(model, dataloader: DataLoader, config):
    epochs = config['epochs']
    loss_fn = torch.nn.NLLLoss()
    optimizer = torch.optim.RMSprop(params=model.parameters(), lr=config['lr'])
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer,
        num_warmup_steps=0, num_training_steps=epochs * len(dataloader)
    )

    # Start training
    model.train()
    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        progress_bar = tqdm(range(len(dataloader)), desc="train", file=sys.stdout)
        tot_loss, tot_samples, tot_corrects = 0, 0, 0

        for batch in dataloader:
            outputs, labels = forward(batch, model)

            # Update model
            loss = loss_fn(torch.log(outputs), labels)
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

            batch_size = labels.size(0)
            predictions = torch.argmax(outputs, dim=-1)
            tot_loss += loss.item() * batch_size
            tot_samples += batch_size
            tot_corrects += (predictions == labels).sum().item()
            clf_metrics.add_batch(predictions=predictions, references=labels)

        progress_bar.close()
        print(f"train metrics: {clf_metrics.compute()}")
        train_loss = tot_loss / len(dataloader)
        train_accuracy = tot_corrects / tot_samples
        val_loss, val_accuracy, _, _ = eval(model, dataloader)
        print(f'epoch: {epoch}, loss: {train_loss:.4f}, accuracy: {train_accuracy:.4f}')
        print(f'epoch: {epoch}, val_loss: {val_loss:.4f}, val_accuracy: {val_accuracy:.4f}')
        print()
    return model


def eval(model, dataloader: DataLoader):
    loss_fn = torch.nn.NLLLoss()
    progress_bar = tqdm(range(len(dataloader)), desc="eval", file=sys.stdout)
    tot_predictions = np.empty(shape=0, dtype=int)
    tot_labels = []
    tot_loss, tot_samples, tot_corrects = 0, 0, 0

    # Start evaluation
    model.eval()
    for batch in dataloader:
        with torch.no_grad():
            outputs, labels = forward(batch, model)

        batch_size = labels.size(0)
        loss = loss_fn(torch.log(outputs), labels)
        tot_loss += loss.item() * batch_size
        tot_samples += batch_size
        predictions = torch.argmax(outputs, dim=-1)
        tot_corrects += (predictions == labels).sum().item()
        tot_predictions = np.concatenate((tot_predictions, predictions.cpu()), axis=None)
        clf_metrics.add_batch(predictions=predictions, references=labels)
        labels = ['true' if l == 0 else 'fake' for l in labels]
        tot_labels = tot_labels + labels
        progress_bar.update(1)

    progress_bar.close()
    print(f"eval metrics: {clf_metrics.compute()}")
    val_loss = tot_loss / len(dataloader)
    accuracy = tot_corrects / tot_samples
    return val_loss, accuracy, tot_predictions, tot_labels


def forward(batch: [], model):
    contents, comments, labels = [], [], []
    for i in range(len(batch)):
        # Move all tensors to device
        cnt = batch[i]['content']
        cnt = {k: v.to(device) for k, v in cnt.items()}
        contents.append(cnt)

        cmt = batch[i]['comment']
        cmt = {k: v.to(device) for k, v in cmt.items()}
        comments.append(cmt)

        labels.append(cnt['labels'])

    labels = torch.LongTensor(labels).to(device)
    outputs = model(contents, comments)
    return outputs, labels


def run(config: {}):
    # Crate save_dir if it doesn't exist
    model_save_dir = config['model_save_dir']
    model_save_path = config['model_save_path']
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    # Multiprocessing
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    df = politifact.get()
    # import pandas as pd
    # df = pd.concat([df.head(n=40), df.tail(n=40)])  # reduce
    df['labels'] = df['fake_news']
    train_set, val_set = politifact.get_splits(df, test_size=0.2)

    # Tokenize dataset
    bert_preset = config['bert_preset']
    tokenizer = AutoTokenizer.from_pretrained(bert_preset)

    print("----- Preprocessing -----")
    if config['encoder'] == 'glove':
        print("----- Loading glove -----")
        glove = Glove(path=GLOVE)
        config['glove'] = glove
        ds_train_content = GloveTextDataset(train_set, config)
        ds_val_content = GloveTextDataset(val_set, config)
    else:
        ds_train_content = BertTextDataset(train_set, config, tokenizer)
        ds_val_content = BertTextDataset(val_set, config, tokenizer)

    collator = SimpleDataCollator()
    batch_size = config['batch_size']
    data_loaders = {
        'train': DataLoader(dataset=ds_train_content, shuffle=False, batch_size=batch_size, collate_fn=collator),
        'val': DataLoader(dataset=ds_val_content, shuffle=False, batch_size=batch_size, collate_fn=collator),
    }

    # print("----- model structure -----")
    model = Defend(config).to(device)
    # print(model)

    if os.path.exists(model_save_path):
        print(f"Loading trained model from: {model_save_path}")
        state_dict = torch.load(model_save_path)
        model.load_state_dict(state_dict=state_dict)
    else:
        print("----- model training -----")
        model = train(model, data_loaders['train'], config=config)
        print(f"Saving trained model from: {model_save_path}")
        torch.save(model.state_dict(), model_save_path)

    print("----- model evaluation -----")
    val_loss, accuracy, predictions, labels = eval(
        model, data_loaders['val']
    )
    print(f'Evaluation: loss: {val_loss:.4f}, accuracy: {accuracy:.4f}')
    metrics.evaluatePerformance(predictions, labels)

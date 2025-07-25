

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from preprocess import ISLGlossDataset, collate_fn
from model import Seq2SeqTransformer
import os


BATCH_SIZE = 2
EPOCHS = 7
LEARNING_RATE = 0.0005
EMB_SIZE = 256 
NHEAD = 4
FFN_HID_DIM = 256
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
MODEL_SAVE_PATH = "models/seq2seq_transformer.pth"


dataset = ISLGlossDataset("data/isl-gloss-dataset.csv")
val_size = int(0.1 * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)


SRC_VOCAB_SIZE = len(dataset.src_vocab)
TGT_VOCAB_SIZE = len(dataset.tgt_vocab)


model = Seq2SeqTransformer(
    num_encoder_layers=NUM_ENCODER_LAYERS,
    num_decoder_layers=NUM_DECODER_LAYERS,
    emb_size=EMB_SIZE,
    src_vocab_size=SRC_VOCAB_SIZE,
    tgt_vocab_size=TGT_VOCAB_SIZE,
    dim_feedforward=FFN_HID_DIM,
    nhead=NHEAD,
    dropout=0.1,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


PAD_IDX = dataset.tgt_vocab['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def calculate_accuracy(preds, labels):
    preds = preds.argmax(-1)
    match = (preds == labels).float()
    mask = (labels != PAD_IDX).float()
    return (match * mask).sum() / mask.sum()

best_val_loss = float('inf')
best_epoch = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    total_acc = 0

    for src, tgt, _, _ in train_loader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        optimizer.zero_grad()
        logits = model(src, tgt_input)

        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += calculate_accuracy(logits, tgt_out).item()

    train_loss = total_loss / len(train_loader)
    train_acc = total_acc / len(train_loader)

   
    model.eval()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for src, tgt, _, _ in val_loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            logits = model(src, tgt_input)

            val_loss += criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1)).item()
            val_acc += calculate_accuracy(logits, tgt_out).item()

    val_loss /= len(val_loader)
    val_acc /= len(val_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

  
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1


os.makedirs("models", exist_ok=True)
torch.save({
    'model_state_dict': model.state_dict(),
    'src_vocab': dataset.src_vocab.stoi,
    'tgt_vocab': dataset.tgt_vocab.stoi
}, MODEL_SAVE_PATH)



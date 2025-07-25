import torch
from model import Seq2SeqTransformer
from preprocess import src_vocab, tgt_vocab


PAD_IDX = src_vocab['<pad>']
BOS_IDX = src_vocab['<bos>']
EOS_IDX = src_vocab['<eos>']


SRC_VOCAB_SIZE = len(src_vocab)
TGT_VOCAB_SIZE = len(tgt_vocab)
EMB_SIZE = 256  
NHEAD = 4
FFN_HID_DIM = 256
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
MAX_LEN = 50

model = Seq2SeqTransformer(
    num_encoder_layers=NUM_ENCODER_LAYERS,
    num_decoder_layers=NUM_DECODER_LAYERS,
    emb_size=EMB_SIZE,
    src_vocab_size=SRC_VOCAB_SIZE,
    tgt_vocab_size=TGT_VOCAB_SIZE,
    dim_feedforward=FFN_HID_DIM,
    nhead=NHEAD,
    dropout=0.1,
    pad_idx=PAD_IDX
)

checkpoint = torch.load("models/seq2seq_transformer.pth", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


def tokenize(text):
    import spacy
    spacy_en = spacy.load("en_core_web_sm")
    return [tok.text.lower() for tok in spacy_en.tokenizer(text)]

def encode(text, vocab):
    tokens = tokenize(text)
    ids = [vocab['<bos>']] + [vocab[token] for token in tokens] + [vocab['<eos>']]
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

def greedy_decode(model, src, max_len, start_symbol):
    src = src.to(torch.device('cpu'))
    memory = model.encode(src, src_mask=None)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long)

    for _ in range(max_len - 1):
        tgt_mask = torch.triu(torch.ones((ys.size(1), ys.size(1))) == 1).transpose(0, 1)
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))

        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        next_word = prob.argmax(dim=1)[0].item() 
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src).fill_(next_word)], dim=1)

        if next_word == EOS_IDX:
            break

    return ys.squeeze(0)



while True:
    sentence = input("\nEnter English sentence: ")
    if sentence.lower() == 'quit':
        break
    try:
        src = encode(sentence, src_vocab)
        output = greedy_decode(model, src, MAX_LEN, BOS_IDX)
        gloss = [tgt_vocab.lookup_token(idx) for idx in output if idx not in [BOS_IDX, EOS_IDX, PAD_IDX]]
        print("\n ISL Gloss:", ' '.join(gloss))
    except Exception as e:
        print(f"\n Error: {e}")

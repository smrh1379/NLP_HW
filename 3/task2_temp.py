import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizerFast, BertForTokenClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
tokenizer = BertTokenizerFast.from_pretrained('dmis-lab/biobert-base-cased-v1.1')


# Function to read file content
def read_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()


# Function to parse annotations from .ann file
def parse_ann(ann_content):
    annotations = []
    for line in ann_content.strip().split('\n'):
        if line.startswith('T'):
            parts = line.split('\t')
            ann_id = parts[0]
            label_info = parts[1]
            text = parts[2]
            label_info_parts = label_info.split()
            label = label_info_parts[0]
            start = int(label_info_parts[1].split(';')[0])
            end = int(label_info_parts[2].split(';')[0])
            annotations.append({
                'id': ann_id,
                'label': label,
                'start': start,
                'end': end,
                'text': text
            })
    return annotations


# Function to preprocess text and remove stop words and punctuation
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words and token not in string.punctuation]
    return filtered_tokens


# Function to format text and annotations for BioBERT input
def format_biobert_input(text, annotations):
    tokens = preprocess_text(text)
    token_annotations = ['O'] * len(tokens)
    text_offset = 0

    for ann in annotations:
        ann_tokens = word_tokenize(ann['text'])
        ann_label = ann['label']

        while text_offset < len(tokens):
            try:
                if tokens[text_offset] == ann_tokens[0]:
                    match = True
                    for i in range(len(ann_tokens)):
                        if text_offset + i >= len(tokens) or tokens[text_offset + i] != ann_tokens[i]:
                            match = False
                            break
                    if match:
                        for i in range(len(ann_tokens)):
                            if i == 0:
                                token_annotations[text_offset + i] = f'B-{ann_label}'
                            else:
                                token_annotations[text_offset + i] = f'I-{ann_label}'
                        text_offset += len(ann_tokens)
                        break
                text_offset += 1
            except:
                print(ann_tokens)
                print(tokens[text_offset])

    return tokens, token_annotations


# Function to split tokens and labels into chunks of up to max_length
def split_into_chunks(tokens, labels, max_length=509):
    chunks = []
    chunk_labels = []
    current_chunk = []
    current_chunk_labels = []
    current_length = 0

    for i in range(len(tokens)):
        current_chunk.append(tokens[i])
        current_chunk_labels.append(labels[i])
        current_length += 1

        if current_length >= max_length:
            # Ensure that we do not split entities and the last label is 'O'
            while i < len(tokens) and not labels[i] == 'O':
                current_chunk.append(tokens[i])
                current_chunk_labels.append(labels[i])
                current_length += 1
                i += 1

            chunks.append(current_chunk)
            chunk_labels.append(current_chunk_labels)
            current_chunk = []
            current_chunk_labels = []
            current_length = 0

    if current_chunk:
        chunks.append(current_chunk)
        chunk_labels.append(current_chunk_labels)

    return chunks, chunk_labels


# Function to process text and annotation files
def process_files(txt_file, ann_file):
    text = read_file(txt_file)
    ann_content = read_file(ann_file)
    annotations = parse_ann(ann_content)
    tokens, labels = format_biobert_input(text, annotations)
    token_chunks, label_chunks = split_into_chunks(tokens, labels)
    return token_chunks, label_chunks


# Lists to store processed tokens and labels
tokend_text = []
cor_labels = []


# Function to process all files in a directory
def process_all_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            txt_file = os.path.join(directory, filename)
            ann_file = txt_file.replace(".txt", ".ann")
            if os.path.exists(ann_file):
                token_chunks, label_chunks = process_files(txt_file, ann_file)
                tokend_text.extend(token_chunks)
                cor_labels.extend(label_chunks)


# Example usage:
directory = 'n2c2/n2c2/part2'
process_all_files(directory)

# Prepare the tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
model = BertForTokenClassification.from_pretrained('dmis-lab/biobert-base-cased-v1.1',
                                                   num_labels=len(set(label for doc in cor_labels for label in doc)))

# Define labels
labels = ["O", "B-Drug", "I-Drug", "B-Strength", "I-Strength", "B-Form", "I-Form", "B-Dosage", "I-Dosage",
          "B-Duration", "I-Duration", "B-Frequency", "I-Frequency", "B-Route", "I-Route", "B-ADE", "I-ADE",
          "B-Reason", "I-Reason"]

# Map labels to IDs
label_map = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label_map.items()}

# Convert tokens and labels to strings for tokenizer
texts = tokend_text
labels = [[label_map[label] for label in doc_labels] for doc_labels in cor_labels]


# Create a custom dataset class
class NERDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        word_labels = self.labels[index]

        encoding = self.tokenizer(text,
                                  truncation=True,
                                  padding='max_length',  # Ensure padding
                                  max_length=self.max_len,
                                  is_split_into_words=True,
                                  return_tensors='pt')

        word_ids = encoding.word_ids(batch_index=0)

        # Create a mask and label array for the tokens
        labels = [-100 if word_id is None else word_labels[word_id] for word_id in word_ids]

        item = {key: val.squeeze() for key, val in encoding.items()}
        item['labels'] = torch.tensor(labels, dtype=torch.long)

        return item


# Prepare the datasets and dataloaders
MAX_LEN = 512
BATCH_SIZE = 16

# Split the data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.1, random_state=42)

train_dataset = NERDataset(train_texts, train_labels, tokenizer, MAX_LEN)
val_dataset = NERDataset(val_texts, val_labels, tokenizer, MAX_LEN)

train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=BATCH_SIZE)
val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=BATCH_SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Custom training loop
def train_model(train_dataloader, val_dataloader):
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    total_steps = len(train_dataloader) * 3
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    model.train()
    for epoch in range(3):
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            model.zero_grad()
            inputs = {key: val.to(device) for key, val in batch.items()}
            outputs = model(**inputs)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        print(f"Epoch {epoch + 1}, Total Loss: {total_loss / len(train_dataloader)}")


def evaluate_model(val_dataloader):
    model.eval()
    true_labels = []
    pred_labels = []

    for batch in val_dataloader:
        inputs = {key: val.to(device) for key, val in batch.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        labels = inputs['labels']

        for i in range(len(labels)):
            true_labels.extend(labels[i].cpu().numpy())
            pred_labels.extend(predictions[i].cpu().numpy())

    true_labels = [label for label in true_labels if label != -100]
    pred_labels = [pred for label, pred in zip(true_labels, pred_labels) if label != -100]

    accuracy = accuracy_score(true_labels, pred_labels)
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(true_labels, pred_labels,
                                                                                 average='micro')
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(true_labels, pred_labels,
                                                                                 average='macro')

    return accuracy, precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro


train_model(train_dataloader, val_dataloader)

accuracy, precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro = evaluate_model(
    val_dataloader)

print(f"Accuracy: {accuracy}")
print(f"Micro Precision: {precision_micro}, Micro Recall: {recall_micro}, Micro F1: {f1_micro}")
print(f"Macro Precision: {precision_macro}, Macro Recall: {recall_macro}, Macro F1: {f1_macro}")

# Save the model after training
model_save_path = './model'
os.makedirs(model_save_path, exist_ok=True)
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

print("\nTraining completed and model saved.")

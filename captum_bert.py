import pandas as pd
import torch
import json
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from collections import defaultdict
import pprint
from torch.utils.data import Dataset
import random
import torch
import torch.nn.functional as F
from captum.attr import LayerIntegratedGradients
from captum.attr import visualization as viz
from captum.attr import LayerConductance
import matplotlib.pyplot as plt


path = "/kaggle/input/proplogic/deepseek_2k_RP_test.parquet"

df = pd.read_parquet(path)

print(df.head())
print(df.shape)


class LogicDataset(Dataset):
    def __init__(self, examples, args=None, simple_tokenizer_vocab=None, tokenizer=None):
        self.simple_tokenizer_vocab = simple_tokenizer_vocab
        if args.keep_only_negative:
            self.examples = [i for i in examples if i["label"] == 0]
        self.examples = examples
        for index, example in enumerate(self.examples):
            self.examples[index] = self.convert_raw_example(example)
        
        random.shuffle(self.examples)
        if args.limit_example_num != -1:
            self.examples = self.examples[:args.limit_example_num]

        # Use the shared tokenizer from main() if provided.
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            # Fallback (kept for standalone use)
            self.tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                do_lower_case=args.do_lower_case,
                cache_dir=args.cache_dir if args.cache_dir else None,
            )
            self.tokenizer.add_special_tokens({"additional_special_tokens": ["[AND]", "[THEN]"]})
        
        self.max_length = args.max_length
        self.args = args
        if args.skip_long_examples:
            self.skip_long_examples()
        
    def __len__(self):
        return len(self.examples)

    def report_length(self):
        all_leng = []
        print("\n\n")
        total = 200
        for example in self.examples:
            leng = " ".join(example["rules"] + example["facts"]).lower() + ' ' + example["query"].lower()
            
            leng = len(self.tokenizer.tokenize(leng))
            all_leng.append(leng)
            if len(all_leng) == total:
                break
        print("Average_length", sum(all_leng) / total)
        print("Max", max(all_leng))
        print("\n\n")
    
    def report_allkinds_of_stats(self):
        print("\n\n")
        # Number of fact percentage
        all = []
        for example in self.examples:
            all.append(len(example["facts"]) / len(example["preds"]))
        print("Number of fact percentage", sum(all) / len(all))

        # Number of rules percentage
        all = []
        for example in self.examples:
            all.append(len(example["rules"]) / len(example["preds"]))
        print("Number of rules percentage", sum(all) / len(all))

    def convert_raw_example(self, example):
        new_example = {}
        new_example["rules"] = []
        for rule in example["rules"]:
            one_rule = ""
            one_rule +=  " [AND] ".join(rule[0])
            one_rule += " [THEN] "
            one_rule += rule[-1]
            one_rule += ' .'
            new_example["rules"].append(one_rule)
        
        new_example["facts"] = []
        for fact in example["facts"]:
            one_fact = "Alice "
            one_fact +=  fact
            one_fact += ""
            new_example["facts"].append(one_fact)
        
        new_example["query"] = "Query: Alice is " + example["query"] + " ?"
        new_example["label"] = example["label"]
        new_example["depth"] = example["depth"]
        new_example["preds"] = example["preds"]
        return new_example


    def __getitem__(self, index):
        
        example = self.examples[index]
        #example = self.convert_raw_example(example)
        
        '''
        "rules": [
        "If Person X is serious, Person Y drop Person X and Person X help Person Y, then Person X get Person Y.",
        "If Person X open Person Y and Person Y help Person X, then Person Y ride Person X."
        ],
        "facts": [
        "Alice is serious.",
        "Alice help Bob.",
        "Bob open Alice."
        ],
        "query": "Alice ride Bob",
        "label": 1
        '''
        if self.args.ignore_fact:
            text_a = " ".join(example["rules"])
        elif self.args.ignore_both:
            text_a = " "
        else:
            text_a = " ".join(example["rules"] + example["facts"])
    
        if self.args.ignore_query:
            text_b = " "
        else:
            text_b = example["query"]
        if self.args.shorten_input:
            text_a = text_a.replace("If", "").replace("then", "")
            text_b = text_b.replace("If", "").replace("then", "")
        return text_a, text_b, example["label"], example

    def collate_fn(self, examples):
        tokenizer = self.tokenizer
        batch_encoding = self.tokenizer(
            [(example[0], example[1]) for example in examples],
            max_length=self.max_length,
            padding="longest",
            truncation=True)
        if "t5" in self.args.model_name_or_path:
            # encode the label as text
            labels_as_text = ["true" if example[2] == 1 else "false" for example in examples]
            target_encoding = self.tokenizer(labels_as_text, padding="longest", max_length=self.max_length, truncation=True)
            label_ids = torch.tensor(target_encoding.input_ids)
            label_ids[label_ids == self.tokenizer.pad_token_id] = -100
        else:
            label_ids = torch.LongTensor([example[2] for example in examples])

        return torch.LongTensor(batch_encoding["input_ids"]), torch.LongTensor(batch_encoding["attention_mask"]), torch.LongTensor(batch_encoding["token_type_ids"]) if 'token_type_ids' in batch_encoding else torch.LongTensor([1]), label_ids, [example[-1] for example in examples]
    
    def skip_long_examples(self):
        keep = []
        counter = 0
        for i in tqdm(range(len(self))):
            example = self[i]
            batch_encoding = self.tokenizer(
            [(example[0], example[1])],
            max_length=self.max_length,
            padding="longest",
            truncation=False)
            if len(batch_encoding["input_ids"][0]) > 650:
                print("Over limit")
                counter += 1
            else:
                keep.append(i)
        print("Skipped ", counter, "out of", len(self))
        self.examples = [self.examples[i] for i in keep]


    def limit_length(self, new_length):
        print("Limiting {} to {}".format(len(self), new_length))
        self.examples = self.examples[:new_length]
    
    @staticmethod
    def split_dataset(file_name):
        all_examples = json.load(open(file_name))
        random.seed(0)
        random.shuffle(all_examples)

        train_examples = all_examples[:len(all_examples) // 10 * 8]
        dev_examples = all_examples[len(all_examples) // 10 * 8:len(all_examples) // 10 * 9]
        test_examples = all_examples[len(all_examples) // 10 * 9:]

        with open(file_name + "_train", "w") as f:
            json.dump(train_examples, f)
        with open(file_name + "_val", "w") as f:
            json.dump(dev_examples, f)
        with open(file_name + "_test", "w") as f:
            json.dump(test_examples, f)

        return
    
    @classmethod
    def initialze_from_file(cls, file, args, tokenizer=None):
        if "," in file:
            files = file.split(",")
        else:
            files = [file]
        all_examples = []
        for file in files:
            with open(file) as f:
                examples = json.load(f)
                all_examples.extend(examples)
        return cls(all_examples, args, tokenizer=tokenizer)
    
    @classmethod
    def initialize_from_file_by_depth(cls, file, args, tokenizer=None):
        examples_by_depth = cls.load_examples_by_depth(file, depth = args.group_by_which_depth)
        datasets_by_depth = {}
        for depth, _data in examples_by_depth.items():
            datasets_by_depth[depth] = cls(_data, args, tokenizer=tokenizer)

        return datasets_by_depth
    
    @staticmethod
    def load_examples_by_depth(file, depth = "depth"):
        with open(file) as f:
            examples = json.load(f)

        examples_by_depth = defaultdict(list)
        for example in examples:
            examples_by_depth[example[depth]].append(example)
        
        return examples_by_depth


class Args:
        model_name_or_path = "OUTPUT/RP/BERT/checkpoint-19"
        # Use the same dir as the model checkpoint so tokenizer & model stay in lockstep
        tokenizer_path = "OUTPUT/RP/BERT"
        val_file_path = "/kaggle/input/logic-dep6-train/prop_examples.balanced_by_backward.max_6.json_train"  # <- use test/val file
        model_type = "bert"
        do_lower_case = True
        eval_batch_size = 16
        max_length = 512
        device = torch.device("cpu")
        n_gpu = 0
        local_rank = -1
        cache_dir = None
        group_by_which_depth = "depth"
        limit_report_depth = -1
        limit_report_max_depth = 6
        # dataset args...
        keep_only_negative = False
        skip_long_examples = False
        limit_example_num = -1
        ignore_fact = False
        ignore_both = False
        ignore_query = False
        shorten_input = False
        shrink_ratio = 1
        further_split = False
        further_further_split = False
        max_depth_during_train = 1000

args = Args()

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained("sunayana981/bert",
    subfolder="BERT)
    # Optional: enforce specials if missing, then resize model later.
specials = ["[AND]", "[THEN]"]
if not all(t in tokenizer.get_vocab() for t in specials):
    print("Special tokens missing in tokenizer; adding them now.")
    tokenizer.add_special_tokens({"additional_special_tokens": specials})


config = AutoConfig.from_pretrained("sunayana981/bert",
    subfolder="BERT/checkpoint-19")
model = AutoModelForSequenceClassification.from_pretrained("sunayana981/bert",
    subfolder="BERT/checkpoint-19", config=config)


if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print(f"Resizing model embeddings from {model.get_input_embeddings().weight.shape[0]} to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

model.to(args.device)
model.eval()

print("\nLoading datasets...")
    # IMPORTANT: pass tokenizer so dataset doesn't create its own
datasets = LogicDataset.initialize_from_file_by_depth(args.val_file_path, args, tokenizer=tokenizer)
depths = sorted(list(datasets.keys()))


eval_dataset = datasets[6]



# pick any idx
idx = 61   



text_a, text_b, label, example = eval_dataset[idx]

print("TEXT A (rules + facts):")
print(text_a)
print("\nTEXT B (query):")
print(text_b)
print("\nGround truth label:", label)



encoding = tokenizer(
    [(text_a, text_b)],
    max_length=args.max_length,
    padding="longest",
    truncation=True,
    return_tensors="pt"
)

input_ids = encoding["input_ids"].to(args.device)
attention_mask = encoding["attention_mask"].to(args.device)

token_type_ids = (
    encoding["token_type_ids"].to(args.device)
    if "token_type_ids" in encoding
    else None
)


actual_vocab_size = model.get_input_embeddings().weight.shape[0]
max_token = input_ids.max().item()

assert max_token < actual_vocab_size, (
    f"Token id {max_token} >= embedding size {actual_vocab_size}"
)


ref_input_ids = torch.full_like(
    input_ids,
    tokenizer.pad_token_id
)


def clf_forward_func(
    input_ids,
    token_type_ids=None,
    position_ids=None,
    attention_mask=None
):
    outputs = model(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        attention_mask=attention_mask
    )
    logits = outputs.logits
    pred_class = torch.argmax(logits, dim=1)
    return logits.gather(1, pred_class.unsqueeze(1)).squeeze(1)



lig = LayerIntegratedGradients(
    clf_forward_func,
    model.get_input_embeddings()   
)


attributions, delta = lig.attribute(
    inputs=input_ids,
    baselines=ref_input_ids,
    additional_forward_args=(
        token_type_ids,
        None,
        attention_mask
    ),
    n_steps=50,
    return_convergence_delta=True
)


token_attributions = attributions.sum(dim=-1).squeeze(0)

token_attributions = token_attributions / torch.norm(token_attributions)

tokens = tokenizer.convert_ids_to_tokens(input_ids[0])


with torch.no_grad():
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids
    ).logits

pred_class = logits.argmax(dim=1).item()
pred_prob = F.softmax(logits, dim=1).max().item()

print("\nPredicted class:", pred_class)
print("Prediction confidence:", pred_prob)
print("IG convergence delta:", delta.item() if delta is not None else None)


vis = viz.VisualizationDataRecord(
    token_attributions,
    pred_prob,
    pred_class,
    pred_class,
    str(pred_class),
    token_attributions.sum().item(),
    tokens,
    delta
)

viz.visualize_text([vis])





# ============================================================
# LayerConductance for [SEP] token
# ============================================================
idx = 61


text_a, text_b, label, example = eval_dataset[idx]

print("\nTEXT A (rules + facts):\n", text_a)
print("\nTEXT B (query):\n", text_b)
print("\nGround truth label:", label)


encoding = tokenizer(
    [(text_a, text_b)],
    max_length=args.max_length,
    padding="longest",
    truncation=True,
    return_tensors="pt"
)

input_ids = encoding["input_ids"].to(args.device)
attention_mask = encoding["attention_mask"].to(args.device)

token_type_ids = (
    encoding["token_type_ids"].to(args.device)
    if "token_type_ids" in encoding
    else None
)

actual_vocab_size = model.get_input_embeddings().weight.shape[0]
max_token = input_ids.max().item()

assert max_token < actual_vocab_size, (
    f"Token id {max_token} >= embedding size {actual_vocab_size}"
)


sep_token_id = tokenizer.sep_token_id
sep_positions = (input_ids[0] == sep_token_id).nonzero(as_tuple=True)[0]

assert len(sep_positions) > 0, "No [SEP] token found!"
sep_pos = sep_positions[0].item()

print("\n[SEP] token position:", sep_pos)


def clf_forward_func(
    input_embeddings,
    token_type_ids=None,
    attention_mask=None
):
    outputs = model(
        inputs_embeds=input_embeddings,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask
    )
    logits = outputs.logits
    pred_class = torch.argmax(logits, dim=1)
    return logits.gather(1, pred_class.unsqueeze(1)).squeeze(1)



embedding_layer = model.get_input_embeddings()

input_embeddings = embedding_layer(input_ids)
baseline_embeddings = torch.zeros_like(input_embeddings)


# ============================================================
# 8. LayerConductance across all layers
# ============================================================
sep_layer_effects = []

num_layers = model.config.num_hidden_layers

for layer_idx in range(num_layers):
    print(f"Processing layer {layer_idx + 1}/{num_layers}")

    layer = model.bert.encoder.layer[layer_idx].output

    lc = LayerConductance(
        clf_forward_func,
        layer
    )

    attributions = lc.attribute(
        inputs=input_embeddings,
        baselines=baseline_embeddings,
        additional_forward_args=(
            token_type_ids,
            attention_mask
        )
    )
    # attributions shape: [1, seq_len, hidden_dim]

    sep_attr = attributions[0, sep_pos, :].abs().sum().item()
    sep_layer_effects.append(sep_attr)


sep_layer_effects = torch.tensor(sep_layer_effects)

if sep_layer_effects.sum() > 0:
    sep_layer_effects_norm = sep_layer_effects / sep_layer_effects.sum()
else:
    sep_layer_effects_norm = sep_layer_effects


with torch.no_grad():
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids
    ).logits

pred_class = logits.argmax(dim=1).item()
pred_prob = F.softmax(logits, dim=1).max().item()

print("\nPredicted class:", pred_class)
print("Prediction confidence:", pred_prob)


plt.figure(figsize=(8, 4))
plt.plot(
    range(1, num_layers + 1),
    sep_layer_effects_norm.cpu().numpy(),
    marker="o"
)
plt.xlabel("Transformer Layer")
plt.ylabel("Normalized [SEP] Contribution")
plt.title("[SEP] Token Contribution Across Layers (LayerConductance)")
plt.grid(True)
plt.show()

for i, val in enumerate(sep_layer_effects_norm):
    print(f"Layer {i+1}: {val.item():.4f}")

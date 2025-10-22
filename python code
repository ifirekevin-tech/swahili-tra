from transformers import AutoTokenizer, AutoModelForCausalLM,pipeline
tokenizer = AutoTokenizer.from_pretrained("C:\\Users\\WESONGA\\Downloads\\ll2")
model = AutoModelForCausalLM.from_pretrained("C:\\Users\\WESONGA\\Downloads\\ll2")
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer
import torch
#from transformers import AutoTokenizer, AutoModelForCausalLM
#Load model and tokenizer
model_name = "gpt2" # Change to the model you want
tokenizer = AutoTokenizer.from_pretrained(model_name)
#Add padding token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#Load the model
model = AutoModelForCausalLM.from_pretrained(model_name)
#Resize the model's token embeddings since we've added a padding token
model.resize_token_embeddings(len(tokenizer))


#Load the dataset
local_file_path = "C:\\Users\\WESONGA\\PycharmProjects\\PythonProject2\\swahi.json"
#ocal_file_path = "C:\\Users\\WESONGA\\PycharmProjects\\pythonProject\\swahili_data.jsonl"
#dataset = load_dataset("json", "C:/Users/WESONGA/PycharmProjects/PythonProject2/swahi.json")
dataset = load_dataset("json", data_files="C:\\Users\\WESONGA\\PycharmProjects\\PythonProject2\\swahi.json")

#Split the dataset into training and validation sets
dataset = dataset["train"].train_test_split(test_size=0.1) # 10% for validation
train_data = dataset["train"]
eval_data = dataset["test"]
#Tokenize the dataset
#def tokenize_function(examples):
    # Change 'examples["text"]' to 'examples["article"]' if needed
    #return tokenizer(examples["articles"], truncation=True)
def tokenize_function(examples):
    # Combine instruction and response into a single text
    texts = [f"Instruction: {inst}\nResponse: {resp}"
             for inst, resp in zip(examples["instruction"], examples["response"])]
    return tokenizer(texts, truncation=True, padding="max_length", max_length=512)

tokenized_train_data = train_data.map(tokenize_function, batched=True)
tokenized_eval_data = eval_data.map(tokenize_function, batched=True)



data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
output_dir="./results", # Directory where the model will be saved
num_train_epochs=3, # Number of epochs
per_device_train_batch_size=4,# Batch size
#evaluation_strategy ="epoch", # Evaluate at the end of each epoch
do_eval=True,
eval_steps=500,
logging_dir='./logs', # Logging directory
save_steps=500, # Save model every 500 steps
fp16=True,
)

trainer = Trainer(
model=model,
args=training_args,
train_dataset=tokenized_train_data,
eval_dataset=tokenized_eval_data, # Eval
data_collator=data_collator,
)
##Check if GPU is availble

print(torch.cuda.is_available())

trainer.train()

eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")
#Save the model and tokenizer locally
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
#Generate text with the model
#generator = pipeline(
 #   "text2text-generation",
  #  model="./fine_tuned_model",
  #  tokenizer="./fine_tuned_model"
#)
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"  # uses GPU if available
)
input_text = "leo ni"
#output =generator(input_text, max_length=50, num_return_sequences=1, truncation=True)
output = generator(input_text,max_new_tokens=50,do_sample=True,top_p=0.9,temperature=0.8)
#Print the generated text
print(output[0]['generated_text'])

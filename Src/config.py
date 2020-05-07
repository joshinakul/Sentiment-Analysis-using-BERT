import transformers

MAX_LEN = 512
BATCH_SIZE_TRAIN = 16
BATCH_SIZE_VALID = 8
EPOCHS = 15
BERT_PATH = "../Data/bert_base_uncased/"
MODEL_PATH = "../Model/model.bin"
TRAINING_CSV = "../Data/imdb.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BERT_PATH,
    do_lower_case = True
)
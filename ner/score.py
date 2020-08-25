# test
sample_text = [
    "Is it true that Jane works at Microsoft?",
    "Joe now lives in Copenhagen."
]
sample_tokens = [x.split() for x in sample_text]

sample_dataset = processor.preprocess(
    text=sample_tokens,
    max_len=MAX_SEQ_LENGTH,
    labels=None,
    label_map=label_map,
    trailing_piece_tag=TRAILING_PIECE_TAG,
)
sample_dataloader = dataloader_from_dataset(
    sample_dataset, batch_size=BATCH_SIZE, num_gpus=None, shuffle=False, distributed=False
)
preds = model.predict(
    test_dataloader=sample_dataloader,
    num_gpus=None,
    verbose=True
)
predicted_labels = model.get_predicted_token_labels(
    predictions=preds,
    label_map=label_map,
    dataset=sample_dataset
)

for i in range(len(sample_text)):
    print("\n", sample_text[i])
    print(pd.DataFrame({"tokens": sample_tokens[i] , "labels":predicted_labels[i]}))

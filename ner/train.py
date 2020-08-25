from ner.prepare import load


train_dataloader, test_dataloader = load()



# # Instantiate a TokenClassifier class for NER using pretrained transformer model
# model = TokenClassifier(
#     model_name=MODEL_NAME,
#     num_labels=len(label_map),
#     cache_dir=CACHE_DIR
# )

# # Fine tune the model using the training dataset
# with Timer() as t:
#     model.fit(
#         train_dataloader=train_dataloader,
#         num_epochs=NUM_TRAIN_EPOCHS,
#         num_gpus=NUM_GPUS,
#         local_rank=-1,
#         weight_decay=0.0,
#         learning_rate=5e-5,
#         adam_epsilon=1e-8,
#         warmup_steps=0,
#         verbose=False,
#         seed=RANDOM_SEED
#     )

# print("Training time : {:.3f} hrs".format(t.interval / 3600))

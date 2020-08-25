from utils import Timer
from prepare import load


train_dataloader, test_dataloader, label_map = load(
    quick_run=False, data_path=DATA_PATH, cache_path=CACHE_DIR, model_name=MODEL_NAME, num_gpus=NUM_GPUS, random_seed=RANDOM_SEED)



with Timer() as t:
    preds = model.predict(
        test_dataloader=test_dataloader,
        num_gpus=None,
        verbose=True
    )

print("Prediction time : {:.3f} hrs".format(t.interval / 3600))



# Get a registered model

true_labels = model.get_true_test_labels(label_map=label_map, dataset=test_dataset)


predicted_labels = model.get_predicted_token_labels(
    predictions=preds,
    label_map=label_map,
    dataset=test_dataset
)

report = classification_report(true_labels,
                               predicted_labels,
                               digits=2
                               )

print(report)

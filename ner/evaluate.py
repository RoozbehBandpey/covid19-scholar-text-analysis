from utils import Timer


with Timer() as t:
    preds = model.predict(
        test_dataloader=test_dataloader,
        num_gpus=None,
        verbose=True
    )

print("Prediction time : {:.3f} hrs".format(t.interval / 3600))


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

from load_data import load_from_raw_data
from sklearn.model_selection import train_test_split

# Load data
train_images, train_labels, test_images, test_ids = load_from_raw_data()


# feature engineering
from feat01 import feat_eng

train_images, train_lables, train_param = feat_eng(train_images, train_labels)
test_images, _, _ = feat_eng(test_images, param=train_param)


# (Cross) Validation
train_images, validation_images, train_labels, validation_labels = train_test_split(
    train_images, train_labels, test_size=0.1, random_state=42
)


# Model workflow
from model01 import model_workflow

val_pred, test_pred = model_workflow(train_images, train_labels, validation_images, validation_labels, test_images)


# Ensemble
from ens import ens_and_decision

pred_binary = ens_and_decision([test_pred], [val_pred], validation_labels)


# Save
with open("submission.csv", "w") as csv_file:
    csv_file.write("id,has_cactus\n")
    for tid, prediction in zip(test_ids, pred_binary):
        csv_file.write(f"{tid},{prediction}\n")

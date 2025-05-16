import os
import warnings
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import optuna
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

def compute_domain_normalization():
    """
    Compute domain-specific normalization parameters (per-channel mean and standard deviation)
    from every image in the /kaggle/input/images folder.
    For each image, the function:
      - Opens the image with proper file handling.
      - Converts it to RGB if needed.
      - Resizes it to 224x224.
      - Converts it to a numpy array and scales pixel values to the [0,1] range.
    Then, it computes the per-channel sum and squared sum on these resized images to derive 
    the accurate mean and standard deviation that reflect the actual input seen by the model.
    """
    base_path = "/kaggle/input"
    images_dir = os.path.join(base_path, "images")
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(".jpg")]

    total_sum = np.zeros(3, dtype=np.float64)
    total_sum_sq = np.zeros(3, dtype=np.float64)
    total_pixels = 0

    for file_name in image_files:
        img_path = os.path.join(images_dir, file_name)
        try:
            with Image.open(img_path) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                # Resize the image to 224x224 before converting to array.
                img = img.resize((224, 224))
                # Convert image to numpy array in [0,1]
                img = np.array(img).astype(np.float32) / 255.0
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            continue

        h, w, _ = img.shape
        total_sum += img.sum(axis=(0, 1))
        total_sum_sq += (img ** 2).sum(axis=(0, 1))
        total_pixels += h * w

    mean_domain = (total_sum / total_pixels).tolist()
    std_domain = np.sqrt((total_sum_sq / total_pixels) - (np.array(mean_domain) ** 2)).tolist()
    print("Computed domain-specific normalization parameters:")
    print("Mean:", mean_domain)
    print("Std:", std_domain)
    return mean_domain, std_domain

def load_data():
    # Load data from /kaggle/input using proper encoding.
    base_path = "/kaggle/input"
    train_path = os.path.join(base_path, "train.csv")
    test_path = os.path.join(base_path, "test.csv")
    sample_sub_path = os.path.join(base_path, "sample_submission.csv")

    train_df = pd.read_csv(train_path, encoding="utf-8")
    test_df = pd.read_csv(test_path, encoding="utf-8")
    sample_sub_df = pd.read_csv(sample_sub_path, encoding="utf-8")

    return train_df, test_df, sample_sub_df

def optimize_dtypes(df, numeric_columns):
    # Downcast numeric columns to save memory.
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce", downcast="float")
    return df

def perform_eda(df):
    # Print the exploratory data analysis (EDA) information in plain text.
    print("=== Start of EDA part ===")
    print("Data Shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head(5))
    print("\nData Types:")
    print(df.dtypes)
    print("\nMissing Values per Column:")
    print(df.isnull().sum())
    print("\nUnique Values per Column:")
    for col in df.columns:
        print(f"{col}: {df[col].nunique()}")
    if "species" in df.columns:
        print("\nTarget Distribution (species):")
        print(df["species"].value_counts())
    print("=== End of EDA part ===")

def preprocess_data(train_df, test_df):
    # Convert target column to a categorical type.
    train_df["species"] = train_df["species"].astype("category")

    # Identify feature columns.
    feature_cols_train = [col for col in train_df.columns if col not in ["id", "species"]]
    feature_cols_test = [col for col in test_df.columns if col != "id"]

    # Optimize data types for memory efficiency.
    train_df = optimize_dtypes(train_df, feature_cols_train)
    test_df = optimize_dtypes(test_df, feature_cols_test)

    # Impute missing values using median imputation.
    for col in feature_cols_train:
        median_val = train_df[col].median()
        train_df[col].fillna(median_val, inplace=True)
    for col in feature_cols_test:
        median_val = test_df[col].median()
        test_df[col].fillna(median_val, inplace=True)

    return train_df, test_df, feature_cols_train, feature_cols_test

def feat_eng(X, y, X_test, train_ids, test_ids):
    """
    Feature engineering: Process tabular features and extract image embeddings.
    This function fine-tunes a pretrained ResNet18 on the leaf images domain and extracts 
    region-adapted embeddings.

    Steps:
      - Compute domain-specific normalization parameters (mean_domain and std_domain).
      - Define transformation pipelines for fine-tuning and inference.
      - Fine-tune a pretrained ResNet18 model on the leaf images.
      - Remove the classification head to obtain image embeddings.
      - Extract embeddings for training and test images.
      - Standard-scale the tabular features and image embeddings separately.

    Returns:
      X_tab_scaled: Tabular features from the train set after scaling.
      image_emb_train_scaled: Image embeddings for training, scaled.
      X_test_tab_scaled: Tabular features from the test set after scaling.
      image_emb_test_scaled: Image embeddings for the test set, scaled.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Compute domain-specific normalization parameters.
    mean_domain, std_domain = compute_domain_normalization()

    # --------------------------------------------------------------------------
    # Step 1: Define transformation pipelines.
    # --------------------------------------------------------------------------
    # Transformation pipeline for fine-tuning with augmentations.
    finetune_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_domain, std=std_domain)
    ])
    # Transformation pipeline for inference (no augmentations).
    inference_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_domain, std=std_domain)
    ])

    # --------------------------------------------------------------------------
    # Step 2: Load and modify pretrained ResNet18 for fine-tuning.
    # --------------------------------------------------------------------------
    model = resnet18(pretrained=True)

    # Freeze all parameters first.
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze parameters in layer3 and layer4.
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace the final fully connected layer with a new one that outputs 99 classes.
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, 99)
    for param in model.fc.parameters():
        param.requires_grad = True

    model = model.to(device)

    # --------------------------------------------------------------------------
    # Step 3: Fine-tune the model on the leaf images.
    # --------------------------------------------------------------------------
    base_path = "/kaggle/input"
    images_dir = os.path.join(base_path, "images")

    class LeafDataset(Dataset):
        def __init__(self, ids, labels, transform):
            self.ids = ids
            self.labels = labels
            self.transform = transform
        def __len__(self):
            return len(self.ids)
        def __getitem__(self, idx):
            img_id = self.ids[idx]
            label = self.labels[idx]
            img_path = os.path.join(images_dir, f"{img_id}.jpg")
            try:
                with Image.open(img_path) as image:
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    image = self.transform(image)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                image = Image.new("RGB", (224, 224))
                image = self.transform(image)
            return image, torch.tensor(int(label), dtype=torch.long)

    train_dataset = LeafDataset(train_ids, y, finetune_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    optimizer = torch.optim.Adam([
        {"params": model.layer3.parameters(), "lr": 1e-5},
        {"params": list(model.layer4.parameters()) + list(model.fc.parameters()), "lr": 1e-4}
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    epochs = 5  # Fine-tuning epochs.

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
        scheduler.step()
        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    # --------------------------------------------------------------------------
    # Step 4: Remove the classification head to obtain embeddings.
    # --------------------------------------------------------------------------
    model.fc = torch.nn.Identity()
    model.eval()

    def extract_embedding(img_id, transform):
        img_path = os.path.join(images_dir, f"{img_id}.jpg")
        try:
            with Image.open(img_path) as image:
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image = transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new("RGB", (224, 224))
            image = transform(image)
        image = image.unsqueeze(0)  # Batch dimension.
        with torch.no_grad():
            embedding = model(image.to(device))
        return embedding.cpu().numpy().flatten()

    # --------------------------------------------------------------------------
    # Step 5: Extract embeddings for training and test images.
    # --------------------------------------------------------------------------
    image_emb_train = []
    for img_id in train_ids:
        emb = extract_embedding(img_id, inference_transform)
        image_emb_train.append(emb)
    image_emb_train = np.array(image_emb_train)

    image_emb_test = []
    for img_id in test_ids:
        emb = extract_embedding(img_id, inference_transform)
        image_emb_test.append(emb)
    image_emb_test = np.array(image_emb_test)

    # --------------------------------------------------------------------------
    # Step 6: Separate and scale the tabular and image features.
    # --------------------------------------------------------------------------
    X_tab = X.values if isinstance(X, pd.DataFrame) else X
    X_test_tab = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
    scaler_tab = StandardScaler()
    X_tab_scaled = scaler_tab.fit_transform(X_tab)
    X_test_tab_scaled = scaler_tab.transform(X_test_tab)

    scaler_emb = StandardScaler()
    image_emb_train_scaled = scaler_emb.fit_transform(image_emb_train)
    image_emb_test_scaled = scaler_emb.transform(image_emb_test)

    # Note: Do not multiply image embeddings by any scaling factor here.
    return X_tab_scaled, image_emb_train_scaled, X_test_tab_scaled, image_emb_test_scaled

def multi_class_log_loss(y_true, y_pred, eps=1e-15):
    # Compute multi-class logarithmic loss.
    n = y_pred.shape[0]
    y_ohe = np.zeros_like(y_pred)
    y_ohe[np.arange(n), y_true] = 1
    y_pred = np.clip(y_pred, eps, 1 - eps)
    y_pred = y_pred / np.sum(y_pred, axis=1, keepdims=True)
    loss = -np.mean(np.sum(y_ohe * np.log(y_pred), axis=1))
    return loss

def main():
    # ----------------------------------
    # Step 1: Data Loading
    # ----------------------------------
    train_df, test_df, sample_sub_df = load_data()

    # ----------------------------------
    # Step 2: Exploratory Data Analysis (EDA)
    # ----------------------------------
    perform_eda(train_df)

    # ----------------------------------
    # Step 3: Data Preprocessing
    # ----------------------------------
    train_df, test_df, feature_cols_train, feature_cols_test = preprocess_data(train_df, test_df)

    # Separate features and target.
    X = train_df[feature_cols_train].copy()
    # Convert species to numeric codes.
    y = train_df["species"].cat.codes.values  
    X_test = test_df[feature_cols_test].copy()

    # Extract ids for matching images (do NOT use sample_submission for test index).
    train_ids = train_df["id"].values
    test_ids = test_df["id"].values

    # ----------------------------------
    # Step 4: Feature Engineering - Extract and scale components once.
    # ----------------------------------
    X_tab_scaled, image_emb_train_scaled, X_test_tab_scaled, image_emb_test_scaled = feat_eng(X, y, X_test, train_ids, test_ids)

    # ----------------------------------
    # Step 5: Bayesian Hyperparameter Optimization with Optuna & Model Training
    # ----------------------------------
    class_counts = np.bincount(y)
    if np.min(class_counts) < 2:
        print("Warning: One or more classes have fewer than 2 samples. Using full training with default hyperparameters.")
        best_params = {
            "max_depth": 10,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "image_emb_scale": 1.0
        }
        # Combine features using default scaling factor.
        X_train_final = np.hstack([X_tab_scaled, image_emb_train_scaled * best_params["image_emb_scale"]])
        X_test_final = np.hstack([X_test_tab_scaled, image_emb_test_scaled * best_params["image_emb_scale"]])
        best_model = XGBClassifier(
            max_depth=best_params["max_depth"],
            learning_rate=best_params["learning_rate"],
            n_estimators=best_params["n_estimators"],
            reg_alpha=best_params["reg_alpha"],
            reg_lambda=best_params["reg_lambda"],
            objective="multi:softprob",
            tree_method="gpu_hist",
            n_jobs=-1,
            random_state=42,
            use_label_encoder=False
        )
        best_model.fit(X_train_final, y)
        train_preds = best_model.predict_proba(X_train_final)
        avg_logloss = multi_class_log_loss(y, train_preds)
        test_preds = best_model.predict_proba(X_test_final)
    else:
        # Define objective function for Optuna.
        def objective(trial):
            image_emb_scale = trial.suggest_float("image_emb_scale", 0.1, 2.0)
            max_depth = trial.suggest_int("max_depth", 3, 15)
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
            n_estimators = trial.suggest_int("n_estimators", 50, 300)
            reg_alpha = trial.suggest_float("reg_alpha", 0.0, 1.0)
            reg_lambda = trial.suggest_float("reg_lambda", 1.0, 10.0)

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            fold_losses = []
            for train_idx, val_idx in skf.split(X_tab_scaled, y):
                # Build training and validation matrices by horizontally concatenating:
                # scaled tabular features and scaled image embeddings (scaled by candidate image_emb_scale).
                X_train_fold = np.hstack([X_tab_scaled[train_idx], image_emb_train_scaled[train_idx] * image_emb_scale])
                X_val_fold = np.hstack([X_tab_scaled[val_idx], image_emb_train_scaled[val_idx] * image_emb_scale])
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]

                model_cv = XGBClassifier(
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    n_estimators=n_estimators,
                    reg_alpha=reg_alpha,
                    reg_lambda=reg_lambda,
                    objective="multi:softprob",
                    tree_method="gpu_hist",
                    n_jobs=-1,
                    random_state=42,
                    use_label_encoder=False
                )
                model_cv.fit(X_train_fold, y_train_fold)
                val_preds = model_cv.predict_proba(X_val_fold)
                loss = multi_class_log_loss(y_val_fold, val_preds)
                fold_losses.append(loss)
            return np.mean(fold_losses)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=20)
        best_params = study.best_trial.params
        avg_logloss = study.best_trial.value
        print(f"Best trial CV Multi-class Log Loss: {avg_logloss}")
        print("Best hyperparameters:", best_params)

        # ----------------------------------
        # Step 6: Final Model Training with the optimal hyperparameters.
        # ----------------------------------
        # Combine full training and test data using the tuned image_emb_scale.
        X_train_final = np.hstack([X_tab_scaled, image_emb_train_scaled * best_params["image_emb_scale"]])
        X_test_final = np.hstack([X_test_tab_scaled, image_emb_test_scaled * best_params["image_emb_scale"]])

        best_model = XGBClassifier(
            max_depth=best_params["max_depth"],
            learning_rate=best_params["learning_rate"],
            n_estimators=best_params["n_estimators"],
            reg_alpha=best_params["reg_alpha"],
            reg_lambda=best_params["reg_lambda"],
            objective="multi:softprob",
            tree_method="gpu_hist",
            n_jobs=-1,
            random_state=42,
            use_label_encoder=False
        )
        best_model.fit(X_train_final, y)
        train_preds = best_model.predict_proba(X_train_final)
        avg_logloss = multi_class_log_loss(y, train_preds)
        test_preds = best_model.predict_proba(X_test_final)

    # ----------------------------------
    # Step 7: Clip and normalize test predictions.
    # ----------------------------------
    test_preds = np.clip(test_preds, 1e-15, 1 - 1e-15)
    test_preds = test_preds / np.sum(test_preds, axis=1, keepdims=True)

    # ----------------------------------
    # Step 8: Save Metric Results in scores.csv
    # ----------------------------------
    scores_df = pd.DataFrame({
        "Multi-class Logarithmic Loss": [avg_logloss, avg_logloss]
    }, index=["xgboost_gbm_tuned", "ensemble"])
    scores_df.index.name = "Model"
    scores_df.to_csv("scores.csv", encoding="utf-8")

    # ----------------------------------
    # Step 9: Generate Submission File
    # ----------------------------------
    # Retrieve candidate species columns from sample_submission.csv (excluding the first 'id').
    species_columns = sample_sub_df.columns.tolist()[1:]
    # The ordering of species in the training data categories:
    train_species_order = train_df["species"].cat.categories.tolist()

    # Create predictions DataFrame using the model probabilities.
    test_preds_df = pd.DataFrame(test_preds, columns=train_species_order)
    # Reorder the DataFrame columns to match the sample submission columns.
    test_preds_df = test_preds_df.reindex(columns=species_columns, fill_value=0)

    submission_df = pd.DataFrame({"id": test_ids})
    submission_df = pd.concat([submission_df, test_preds_df], axis=1)
    submission_df.to_csv("submission.csv", index=False, encoding="utf-8")

    print("Scores and submission files generated successfully.")

if __name__ == "__main__":
    main()
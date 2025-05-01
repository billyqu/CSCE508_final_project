# CSCE508 Final Project

This repository contains the code, models, and dataset instructions for our CSCE508 final project, which explores multi-channel training and evaluation using various deep learning models.

---

## 📁 Dataset & Model Downloads

- 📥 **Dataset Download:** [https://drive.google.com/drive/folders/1vMu-ZUzGJgz_3B8YSoYZetXCyyKMTzry?usp=sharing]
- 📥 **Trained Model Download:** [https://drive.google.com/file/d/1VhGYnDe9UlWmjOD7vWNmyXHn6I-NJBDC/view?usp=sharing]

> **Note:** After downloading the dataset, update the dataset paths in the following files:
- `train_multi_channel.py`: Line 193–195
- `test_multi_channel.py`: Line 723–725

---

## 🧠 Model Training

To train a model, use the following command:

```bash
python train_multi_channel.py --model [unet | efnet | swinv2 | coatnet | eva | mamba | resnet | vit | rs | resmlp] --epochs [number_of_epochs]
```

> `--epochs` defaults to 50 if not specified.

---

## 🧪 Model Testing

To test a trained model, use:

```bash
python test_multi_channel.py --model [unet | efnet | swinv2 | coatnet | eva | mamba | resnet | vit | rs | resmlp] --model_path [path_to_saved_model]
```

Replace `[path_to_saved_model]` with your actual file path.

---

## 👥 Contribution

- **Yousef** and **Yueling** trained and tested three models each.
- **Billy** trained and tested four models.
- **Yousef** handled data preprocessing for the classification task.
- **Yueling** conducted research on alternative datasets and model selection strategies.
- All team members contributed to writing the final report, each focusing on their assigned tasks.

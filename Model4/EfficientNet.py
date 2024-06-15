from fastai.vision.all import *
from fastai.callback.all import *
from fastai.vision.models import efficientnet_b0
from fastai.metrics import *
import numpy as np
import seaborn as sns
from sklearn.metrics import *
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# # Step 1: Define the path to your dataset
# path = Path('/Users/visheshgoyal/Python Projects/Leaf Project/MangoLeafDBAryaShah')

# Check if the path exists
# if not path.exists():
#     print(f"Error: The path {path} does not exist.")
# else:
#     print(f"Dataset path: {path}")

# # Step 2: Create the DataLoaders with transformations
# dls = ImageDataLoaders.from_folder(
#     path, 
#     valid_pct=0.2,  # Use 20% of data for validation
#     item_tfms=Resize(448),  # Resize images to 448x448
#     batch_tfms=aug_transforms(size=224, max_warp=0)  # Data augmentations
# )

# # Step 3: Show length of training and testing data to verify
# print(f'{len(dls.train_ds)}')
# print(f'{len(dls.valid_ds)}')

# # Step 4: Create the learner with EfficientNet architecture
# learn = cnn_learner(dls, efficientnet_b0, metrics=[error_rate, accuracy, Precision()])

# # Step 5: Train the model
# learn.fine_tune(5)  # Fine-tune for 5 epochs

# # Step 6: Save the model
# # learn.save('efficientnet-model')
# learn.export('/Users/visheshgoyal/Python Projects/Leaf Project/Mangoleaf/Model4/Trained Model/EfficentNetFinal.pkl')

# Step 7: Import the model
learn = load_learner('/Users/visheshgoyal/Python Projects/Leaf Project/Mangoleaf/Model4/Trained Model/EfficentNetFinal.pkl')

# Step 8: Load a new image and predict its class
path = '/Users/visheshgoyal/Desktop/Unknown.jpeg'
img = PILImage.create(path)
pred_class, pred_idx, probs = learn.predict(img)
probs *= 100
print(f'Prediction: {pred_class}, Probability: {probs[pred_idx]:.4f}%')

# # Step 9: Calculating metrics and confusion matrix
# preds, targs = learn.get_preds(dl=dls.valid)
# pred_labels = preds.argmax(dim=1)
# pred_labels = pred_labels.numpy()
# targs = targs.numpy()
# f1 = f1_score(targs, pred_labels, average='macro')
# acc = accuracy_score(targs, pred_labels)
# precision = precision_score(targs, pred_labels, average='macro')
# print(f"F1 Score: {f1}")
# print(f'Validation Accuracy: {acc:.4f}')
# print(f'Precision Score: {precision:.4f}')

# # Confusion Matrix
# class_labels = ["Healthy","Infected"] 
# cm = confusion_matrix(targs, pred_labels, labels=np.arange(len(class_labels)))

# # Convert confusion matrix to a DataFrame for visualization
# cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
# plt.figure(figsize=(10, 7))
# sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='g')
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()
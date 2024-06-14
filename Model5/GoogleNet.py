from fastai.vision.all import *
from torchvision.models import googlenet
from pathlib import Path
from fastai.metrics import error_rate
from sklearn.metrics import *

# Path to your dataset
path = Path('/Users/visheshgoyal/Python Projects/Leaf Project/MangoLeafDBAryaShah')

# Load the data
dls = ImageDataLoaders.from_folder(
    path,
    valid_pct = 0.2,
    item_tfms=Resize(448),
    batch_tfms=aug_transforms(mult=2)
)

# Load the GoogLeNet model
googlenet_model = googlenet(pretrained=True, aux_logits=True)

# Check if the model is created correctly2
if googlenet_model is None:
    raise ValueError("GoogLeNet model is not instantiated correctly.")

# Remove the auxiliary outputs to fit fastai's expectation of a single output
googlenet_model.aux_logits = False

# Create a CNN learner
learn = Learner(dls, googlenet_model, metrics=error_rate)

# Check if the learner is created correctly
if learn is None:
    raise ValueError("Learner is not created correctly.")

learn.load('/Users/visheshgoyal/Python Projects/Leaf Project/MangoLeafDBAryaShah/models/Googlenet-2-finetune')

def learn_in_interval(modelVar, epochNum, modelPath):

    for epoch in range(0, epochNum - 1):
        modelVar.fine_tune(1)
        modelVar.save(modelPath)
        user_input = input("Enter continue or stop : ")
        if user_input.lower() == 'continue':
            print(f"Training continued. Number of epoch done till now : {epoch}")
        else :
            print(f"Training interrupted. Epochs done : {epoch}")
            return modelVar

    return modelVar

learn = learn_in_interval(modelVar=learn, epochNum=2, modelPath='/Users/visheshgoyal/Python Projects/Leaf Project/MangoLeafDBAryaShah/models/Googlenet-2-finetune.pth')

# Export the learner for inference later
learn.export('/Users/visheshgoyal/Python Projects/Leaf Project/Mangoleaf/Model5/Trained Model/GoogleNetFinal2.pkl')

# # Load Model
# learn = load_learner('/Users/visheshgoyal/Python Projects/Leaf Project/Mangoleaf/Model5/Trained Model/GoogleNetFinal.pkl')

# # Load image and test
# img = PILImage.create('')
# pred, pred_idx, probs = googleNetModel.predict(img)

# print(f'Prediction: {pred}, Probability: {probs[pred_idx]:.4f}')

# # Calculating metrics
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
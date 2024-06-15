Models required:
    - Fast.AI
    - Seaborn
    - Math Plot Lib

Brief:
    - Hey guys, this repository has different image detection models. 
    - The models are trained on a database of 8000 images of both healthy and infected mango plant leaves
    - When you add a new mango leaf image, the models are capable of detecting the disease with about 98% accuracy(Different for each model)
    - Fast.AI framework has been used quite extensively. 
    - Models are: 
        - The Resnet18 Model is the least effective one with accuracy of just 84%. 
        - GoogleNet model has highest 99% accuracy but takes very long time to get trained on the dataset, about 6-7 hours per epoch
        - EfficientNet model has 98% accuracy and takes just 57 minutes for one epoch on our database. 
    - Python has been used. 

The models in Trained Model folder with .pkl extension are ready to use models. 

The commented lines are to be used as and when required for calculating metrics, adding new images, training the models etc but every line is important and has its role mentioned in the previous line. 
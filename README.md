Models required:<br/>
    - Fast.AI<br/>
    - Seaborn<br/>
    - Math Plot Lib<br/><br/>

Brief:<br/>
    - Hey guys, this repository has different image detection models. <br/>
    - The models are trained on a database of 8000 images of both healthy and infected mango plant leaves<br/>
    - When you add a new mango leaf image, the models are capable of detecting the disease with about 98% accuracy(Different for each model)<br/>
    - Fast.AI framework has been used quite extensively. <br/>
    - Models are: <br/>
        - The Resnet18 Model is the least effective one with accuracy of just 84%. <br/>
        - GoogleNet model has highest 99% accuracy but takes very long time to get trained on the dataset, about 6-7 hours per epoch<br/>
        - EfficientNet model has 98% accuracy and takes just 57 minutes for one epoch on our database. <br/>
    - Python has been used. <br/><br/>

The models in Trained Model folder with .pkl extension are ready to use models. <br/><br/>

The commented lines are to be used as and when required for calculating metrics, adding new images, training the models etc but every line is important and has its role mentioned in the previous line. <br/><br/>

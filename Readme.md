# AWS SageMaker

# TOC

- [AWS SageMaker](#aws-sagemaker)
- [TOC](#toc)
- [Assignment](#assignment)
- [Solution](#solution)

# Assignment

* https://www.kaggle.com/datasets/puneet6060/intel-image-classification dataset used
* Use **Custom Docker Image** for all Sagemaker Jobs
  * Keep the Inference Container light weight
  * You don't need to create the image from scratch, rather just take one of the AWS DLC Image as Base
  * You will need these examples
    * https://docs.aws.amazon.com/sagemaker/latest/dg/build-your-own-processing-container.html
    * https://github.com/aws/amazon-sagemaker-examples/blob/main/advanced_functionality/pytorch_extending_our_containers/pytorch_extending_our_containers.ipynb
* Pre-Process dataset
  * Dataset: https://www.kaggle.com/datasets/puneet6060/intel-image-classification
    * Your dataset is already split into train and test
    * Resize the images to your model input size
    * Version the dataset with DVC
    * Push to S3
* Train Resnet18 from TIMM (Pretrained)
  * Use GPU Training
    * **Single GPU, Single Node**
* Save model evaluation metrics to a json file in sagemaker model directory
  * this would get uploaded to S3 automatically along with model artifacts
* Deploy Model to SageMaker inference endpoint
  * Resizing and Standardization of Input Image **must be in your inference script.**
  * You will only send the image as an array to the endpoint
  * **The response must be a dict of top 5 predictions along with confidence**
* Test endpoint with **2 example images each** from the classnames in the dataset
  * images can be taken from pred folder of the dataset, or from google images
* **Share the Notebook** with the above endpoint inference in notebook
  * It should include the preprocessing, training, deployment logs
  * It should also include the example images for testing along with their predictions
* Submit your training, preprocessing and inference scripts uploaded to Github
* Upload Tensborboard logs to Tensorboard Dev and share link to it

# Solution

- Custom Docker Image
  <details>
  <summary><b>Dockerfile:</b></summary>

  ```Dockerfile
  FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.12.1-gpu-py38-cu113-ubuntu20.04-sagemaker

  COPY requirements.txt .
  RUN pip3 install -r requirements.txt

  ```
- [preprocessing data script](preprocess.py), [training model script](train.py) & [infrencing script](infer.py)
- [setup dvc and git](notebooks/01-setup-git-dvc.ipynb) | [building pipeline](notebooks/02-pipeline.ipynb)
- [tensorboard dev logs](https://tensorboard.dev/experiment/BxMWktWVTyuOoLOf9ydTPA/)
  
- Pipeline

![](./images/pipeline.png)
  ```bash
  
  model evaluation

  :: Eval Metrices->  { "accuracy": {
                            "value": 0.9256359934806824,
                            "per_class_accuracy": {
                                "buildings": 0.946835458278656,
                                "forest": 0.9854369163513184,
                                "glacier": 0.8374717831611633,
                                "mountain": 0.9252747297286987,
                                "sea": 0.980861246585846,
                                "street": 0.8865740895271301
                            },
                            "standard_deviation": 0.057204633951187134
                        },
                        "loss": 0.21532145142555237,
                        "f1_score": -0.9256359934806824,
                        "precission_micro": 0.9256359934806824,
                        "precission_macro": 0.9274089932441711,
                        "precission_weighted": 0.926857590675354,
                        "recall": 0.9256359934806824
                    }
    confusion_matrix = 
        [[374,   0,   0,   2,   2,  17],
        [  0, 406,   0,   6,   0,   0],
        [  1,   2, 371,  49,  17,   3],
        [  0,   1,  27, 421,   6,   0],
        [  0,   1,   1,   6, 410,   0],
        [ 42,   1,   0,   0,   6, 383]]
  ```

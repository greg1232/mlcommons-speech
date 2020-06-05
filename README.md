# mlcommons-speech
Simple test code for speech dataset 

# to install
```
  python3 -m venv environment
  source environment/bin/activate
  
  pip install -r requirements.txt
```
  
# to run

```
  python train.py -m config/sota.json 
```

# helpful arguments

```
  AWS_REGION=us-east-2 TF_CPP_MIN_LOG_LEVEL=3 python train.py -m config/sota.json 
```

* Sets the correct AWS region
* Supressed tensorflow S3 logs

# Principles

* Push button:  It should be possible to clone this repo and reproduce the SOTA result on multiple machines with one short command.
* A good model: The SOTA model should have a short and clean implementation and achieve high accuracy.



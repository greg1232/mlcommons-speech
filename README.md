# mlcommons-speech
Simple test code for speech dataset testing

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



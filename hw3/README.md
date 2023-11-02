## Set up environment
We recommend you to set up a conda environment for packages used in this homework.
```
conda create -n 2590-hw3 python=3.8
conda activate 2590-hw3
pip install -r requirements.txt
```

After this, you will need to install certain packages in nltk
```
python3
>>> import nltk
>>> nltk.download('wordnet')
>>> nltk.download(’punkt’)
>>> exit()
```
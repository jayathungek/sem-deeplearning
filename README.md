# SEM Lab Session

## Environment setup
```console
conda env create -f environment.yml
conda activate sem-deep-learning
mkdir model_dir saved_models
```

## Getting datasets
Use this to host the files online: https://www.station307.com/#/ - will make the files available to stream from host machine over network until browser tab is closed, very handy in a classroom setting.
### Metmusem 
```console
mkdir -p data/metmuseum
wget <station307-url> -O data/metmuseum/MetObjects.csv
```
### MNIST
1. Download the [MNIST as \.jpg dataset](https://www.kaggle.com/datasets/scolianni/mnistasjpg) from the station307 url  
```console
wget <station307-url> -O mnist.zip
```
2. Extract to data dir
```console
mkdir -p data/MNIST/raw
unzip mnist.zip -d data/MNIST
mv data/MNIST/trainingSet/trainingSet/* data/MNIST/raw
```
3. Remove unneeded files
```console
rm -r mnist.zip data/MNIST/testS* data/MNIST/trainingS* 
```
4. Create filepath to label mapping:
```console
cd data/MNIST && chmod +x create_csv.sh
./create_csv.sh raw
```


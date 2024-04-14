URL=https://www.dropbox.com/s/d1kjpkqklf0uw77/celeba.zip?dl=0
ZIP_FILE=./drive/MyDrive/StarGAN/data/celeba.zip
wget -N $URL -O $ZIP_FILE
unzip $ZIP_FILE -d ./drive/MyDrive/StarGAN/data/
rm $ZIP_FILE

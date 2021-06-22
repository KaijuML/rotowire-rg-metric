cd data

mkdir rotowire
mkdir rotowire/json
mkdir rotowire/models
mkdir rotowire/output
mkdir rotowire/gens

git clone https://github.com/harvardnlp/boxscore-data

cd boxscore-data

tar -xvf rotowire.tar.bz2

mv rotowire/* ../rotowire/json

rm -rf ../boxscore-data/
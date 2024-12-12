set dotenv-load

default: info

# Create conda environment (cpu version)
create_env_cpu:
    conda env create -f ./env/environment_cpu.yml

# Create conda environment (gpu version)
create_env_gpu:
    conda env create -f ./env/environment_gpu.yml

# Download AFEW SPD dataset
download_afewspd:
    wget https://data.vision.ee.ethz.ch/zzhiwu/ManifoldNetData/SPDData/AFEW_SPD_data.zip -P ./data/
    unzip ./data/AFEW_SPD_data.zip -d ./data/AFEW_spdnet/
    rm ./data/AFEW_SPD_data.zip

# Download SP_marti
download_spmarti:
    wget https://sp500-histo.s3-ap-southeast-1.amazonaws.com/CorrMats.zip  -P ./data
    unzip ./data/CorrMats.zip -d ./data/SP_marti/
    rm ./data/CorrMats.zip


# Download HDM05 SPD dataset
download_hdm05spd:
    wget https://data.vision.ee.ethz.ch/zzhiwu/ManifoldNetData/SPDData/HDM05_SPDData.zip -P ./data
    unzip ./data/HDM05_SPDData.zip -d ./data/HDM05_spdnet
    rm ./data/HDM05_SPDData.zip

# Run SPDnet training on AFEW dataset
run_afew:
    mkdir -p results/afew
    python ./experiments/train_afew.py --storage_path ./results/afew/ 

# Run SPDnet training on AFEW dataset - Kobler implementation
run_afew_kobler:
    mkdir -p results/afew
    python ./experiments/train_afew_kobler.py --storage_path ./results/afew_kobler/

# Run SPDNet training on SP_marti dataset
run_spmarti:
    mkdir -p results/sp_marti
    python ./experiments/train_spmarti.py --storage_path ./results/sp_marti

# Run SPDNet training on SP_marti dataset
run_hdm05:
    mkdir -p results/hdm05
    python ./experiments/train_hdm05.py --storage_path ./results/hdm05

# Download anotherspdnet from Github
get_anotherspdnet:
    git submodule update

# Install anotherspdnet from local files
install_anotherspdnet:
    pip uninstall anotherspdnet -y
    cd ./anotherspdnet; pip install .

info:
    @echo "$PROJECT_NAME : $PROJECT_DESCRIPTION\nBy $AUTHOR - $CONTACT"

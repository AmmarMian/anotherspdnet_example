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

# Run SPDnet training on AFEW dataset
run_afew:
    mkdir -p results/afew
    python ./experiments/train_afew.py --storage_path ./results/afew/ 

# Rub SPDnet training on AFEW dataset - Kobler implementation
run_afew_kobler:
    mkdir -p results/afew
    python ./experiments/train_afew_kobler.py --storage_path ./results/afew/


# Download anotherspdnet from Github
get_anotherspdnet:
    git submodule update

# Install anotherspdnet from local files
install_anotherspdnet:
    pip uninstall anotherspdnet -y
    cd ./anotherspdnet; pip install .

info:
    @echo "$PROJECT_NAME : $PROJECT_DESCRIPTION\nBy $AUTHOR - $CONTACT"

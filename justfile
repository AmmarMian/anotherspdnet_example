set dotenv-load

default: info


# Create conda environment (cpu version)
create_env_cpu:
    conda env create -f ./env/environment_cpu.yml

# Create conda environment (gpu version)
create_env_gpu:
    conda env create -f ./env/environment_gpu.yml

info:
    @echo "$PROJECT_NAME : $PROJECT_DESCRIPTION\nBy $AUTHOR - $CONTACT"

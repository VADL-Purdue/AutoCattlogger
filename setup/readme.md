# Instructions to setup the environment
Author: Manu Ramesh

---

## Get the AutoCattlogger
- Clone the repo and update submodules (recursively clone all dependencies)
    ```
    git clone --recursive https://github.com/VADL-Purdue/AutoCattlogger.git 
    ```
Follow one of the two approaches below - using dcoker or using our conda environment.

## Docker: (verified, works)

Steps:
- Download and isntall docker if your system doesn't have it.
- Install nvidia-container-toolkit to allow docker containers to access gpus.
    - Instructions are in the [nvidia-container-toolkit installation guide page](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
    - Link to [video guide ](https://www.youtube.com/watch?v=KlTz3SJWnVk).
- Go to the docker directory and build the docker image.
    ```
    cd AutoCattlogger/setup/docker/
    docker build -t AutoCattloggerImg:<tag> . 
    
    #replace <tag> with any tag you wish to provide, remove it with the preceeding colon
    # You can also replace the name AutoCattloggerImg with any name you wish to provide.
    # If you get an error saying that you do not have permission to run the above command, try it with sudo.
    ```
- Verify that the image is built with `sudo docker image ls` command. The image must be listed in the output.
- Run the image to get the docker container. Run it in interactive mode using the following command.
    ```
    sudo docker run -it --gpus all -d -p 5000:5000 -v ../:/AutoCattlogger/ --shm-size=64gb AutoCattloggerImg
    ```
    You can change the shared memory size 'shm-size' depending on the amount of RAM available on your system. The default of 64MB is too less to run model training.

    
    You could also grant the docker container access to directories on your computer by mapping them into locations inside the container using the -v option in the run command.
    This is useful if you wish to share directories with data or models with the container.
    Note that you can change the mount location to any location you want by replacing the "/mnt/share/" string in the command below.
    ```
    sudo docker run -it --gpus all -d -p 5000:5000 -v <path/to/local-dir-you-want-to-share/>:/mnt/share/ mmpose-cu-11.6-pt1.13.0-new
    ```
    For now, if you use the first command above, it will mount the AutoCattlogger directory at /AutoCattlogger/ in the container. So, any data or models you save inside your AutoCattlogger directory should be mapped here.
- Verify that the container is running using the `sudo docker ps` command and note down the container id.
- Run the container in interactive mode. Replace the \<container-id\> below with the container id that you noted down earlier.
    ```
    sudo docker exec -it <container-id> /bin/bash
    ```
- Once in the container shell,
    - Move into the AutoCattlogger directory 
        ```
        cd ../AutoCattlogger/
        ```
    - Start using the AutoCattlogger by following instructions in the [main readme file](../README.md).

### Extras
- To make the container run in the background, press `ctrl+p` followed by `ctrl+q`.
- Get back to the interactive shell using `sudo docker attach <container_id_or_name>`.

- To quit the container, run the `exit` command.
- To stop the container, run `sudo docker stop <container-id>`. Verify by running `sudo docker ps`.
- To delete the container image altogether, run `sudo docker rmi -f AutoCattloggerImg`. Verify by running `sudo docker image ls`.


## Conda environment

### Option1: Build a new env from yaml file (Has bugs, might not work, pull-requests with fixes are welcome)

- Use the provided yaml file "autoCattlogger.yaml" to create the conda env.
    ```
    cd AutoCattlogger/setup/
    conda env create -f autoCattlogger.yaml

    # activate the environment
    conda activate autoCattlogger
    ```

#### Extras
- Install dependencies
    ```
    cd ../dependencies/
    
    python -m pip install torch torchvision -U

    # detectron2
    python -m pip install -e detectron2

    #mmpose - follow installation instructions instructions from here: https://mmpose.readthedocs.io/en/latest/installation.html#install-as-a-python-package

    python -m pip install -U openmim
    mim install mmengine
    mim install "mmcv>=2.0.1"
    mim install "mmdet>=3.1.0"
    python -m pip install -v -e mmpose
    ```

- We also provide a requirements.txt file in case you wish to directly install pip dependencies.
    ```
    python -m pip install -r requirements.txt
    ```

### Option2: Copy and run our ready-made conda env (works for us)
- Download the conda environment from [here](https://app.box.com/s/zdxwo7cc34fzjbvd26tqj3aqy1r5laeh).
    - For downloading it using cli, run this command:
        ```
        curl -L https://app.box.com/shared/static/zdxwo7cc34fzjbvd26tqj3aqy1r5laeh --output det2_openmmlab_condaEnv_for_AutoCattlogger.tar
        ```
    - Note that this is a 5.2GB file.
- Extract this folder and copy it to your anaconda3/envs/ directory.
    - Use `tar -xvf det2_openmmlab_condaEnv_for_AutoCattlogger.tar` to extract the environment directory.
    - The *det2_openmmlab* folder inside the det2_openmmlab_condaEnv_for_AutoCattlogger folder that is extracted must be copied into the anaconda3 envs folder.
- The path to your anaconda installation directory can be found using `which anaconda` command.
- The environment is named *det2_openmmlab* as it supports detectron2 and openmmlab (for MMPose). You can rename it to AutoCattlogger if you would like to.
    - You can also use any environment that supports both these frameworks.
- Run `conda env list` command to verify that the env is registered.
- Run `conda activate <env-name>` to activate and start using the environment.




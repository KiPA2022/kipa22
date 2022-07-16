# Docker submission guidelines

In closed test, we require the participants to build a docker image of their algorithm, and upload it to docker hub. 

We will pull your docker image and run it on our server, get your inference results and then evaluate them using 12 metrics (as same as the open test). 

This dummy submission shows how to build a docker image of a baseline model (DenseBiasNet). The `Dockerfile`, `predict.py` and `requirements.txt` are added to the directory, other files are all copied from `../BASELINES/`. To get familiar with docker, you can download this directory, and run the command `$ docker build -t dummy .` to get a dummy docker image.  

## Testing environment

* nVidia GeForce RTX 3090 (**your packages should be compatible with CUDA capability of sm_86**)
* CUDA 11.1 (Recommend torch>=1.10)
* 24GB GPU memory

## If you are familiar with docker...

1. Make sure your algorithm reads images from `/input`, and predicted files (with the same filename as corresponding images) are written to `/output`. 
2. Build and save your docker image. Note that it should be compatible with our hardware platform. 
3. Upload your docker image to [docker hub](https://hub.docker.com/). Then [Email](KiPA2022@outlook.com) us the tag. 
4. We will pull and run your docker image using the following commands:  
`$ docker pull YOUR_USER_NAME/YOUR_DOCKER_IMAGE_NAME`  
`$ docker run --rm --gpus all --runtime=nvidia --ipc=host -v LOCAL_INPUT:/input/:ro -v LOCAL_OUTPUT:/output/ YOUR_DOCKER_IMAGE_NAME`  
This might help you do some sanity checks (if this command runs successfully on your local machine then it might works on our machine as well). 

## If you are not familiar with docker...

1. Download and install [docker](https://docs.docker.com/engine/install/) on your machine. You need to register for an account in advance if you don't have one. 
2. Copy the [`Dockerfile`](Dockerfile) in this repository to your code's top directory. 
3. Make sure your algorithm reads images from `/input`, and predicted files (with the same filename as corresponding images) are written to `/output`. If necessary, create a new python script to run the inference only, like [`predict.py`](predict.py). 
4. Save your environment package list to a `requirements.txt` using   
`$ pip freeze > requirements.txt`   
or   
`$ conda list -e > requirements.txt`  
NOTE: This step is recommanded, you can also configure your environment manually in Dockerfile (in that case, replace the line `pip install -r requirements.txt` with `RUN pip install ...` ). 
5. Edit [`Dockerfile`](Dockerfile) from line 18 to line 25, these lines instruct which files should be copied to the docker image. So, replace them with the files that you need to copy. 
6. Build your docker image using  
`$ docker build -t YOUR_DOCKER_IMAGE_NAME .`  
Check if it runs normally on your local machine using  
`$ docker run --rm --gpus all --runtime=nvidia --ipc=host -v LOCAL_INPUT:/input/:ro -v LOCAL_OUTPUT:/output/ YOUR_DOCKER_IMAGE_NAME`  
Explanitions: 
    - This command creates a "container" for your image. The container runs on your local machine, which is "host". 
    - `--runtime=nvidia`: your docker image should have access to nVidia GPUs, so you might need to install [nvidia-docker2](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) preliminarily. 
    - `-v LOCAL_INPUT:/input/:ro`: this flag maps your host path to container path, `ro` means read-only. Your code reads images from `/input`, so the container path should be `/input`. Put some images (for test) in some local path, and that is `LOCAL_INPUT`.   
7. Save your docker image using  
`$ docker save YOUR_DOCKER_IMAGE_NAME | gzip -c > TAR_FILENAME.tar.gz`  
for some Windows users this might doesn't work, use these 2 commands instead:  
`$ docker save -o TAR_FILENAME.tar YOUR_DOCKER_IMAGE_NAME `  
`$ gzip TAR_FILENAME.tar`  
Check if it can be loaded normally using  
`$ docker load -i TAR_FILENAME.tar.gz`  
NOTE: This step is also recommended, pushing your docker image to the hub doesn't require saving it locally. But we still recommend doing this because you can send this docker image to us directly in case you cannot push it to the hub or we cannot pull it somehow. 
8. Push your docker image to docker hub. Use the command  
`$ docker push YOUR_DOCKER_IMAGE_NAME`  
You can also push it using Docker Desktop GUI.  
You might need to re-tag your docker image in advance (docker hub only accepts images with user name), using  
`$ docker tag YOUR_DOCKER_IMAGE_NAME YOUR_USER_NAME/YOUR_DOCKER_IMAGE_NAME`  
After successful pushing, you will be able to find it on the docker hub website. Check if you can pull it using  
`$ docker pull YOUR_USER_NAME/YOUR_DOCKER_IMAGE_NAME`  
9. Email us your tag (`YOUR_USER_NAME/YOUR_DOCKER_IMAGE_NAME`). 

## Troubleshooting

TODO

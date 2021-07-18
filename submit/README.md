# Making dockerfile and implement `inference.py` 
> docker image build guideline (assumed after the docker installation and launching)

1. Build the Docker image: `docker build -t <docker image name> -f submit/Dockerfile .` 
   1. ex> `docker build -t docker_test -f submit/Dockerfile .` 
   2. Your working directory is the upper of `submit` folder
2. Generate the Docker container: `docker run --gpus all -it <docker image name>` 
   1. ex> `docker run --gpus all -it docker_test`
   2. Your working directory is the upper of `submit` folder.
   3. Once you successfully run, your working env will be changed to `root@something:/submit#`
3. Run the `inference.py`: `python inference.py`
   1. Please wait and pray for the future.
4. Check the ouput folder: `cd output/`, and then `ls` to check the result (image, gif, and mp4 files)
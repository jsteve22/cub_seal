# Cub Seal

A repository to hold all of the files to demo new HE convolutional layer evaluation. 

#### Building 

1. Clone the Cub Seal git repository by running:
    ```
    git clone https://github.com/jsteve22/cub_seal.git
    ```

2. Enter the Framework directory: `cd cub_seal/`

3. With docker installed, run:
    ```
    docker build -t cub_seal .
    docker run -it cub_seal --name cub
    ```

4. To re-run the docker image:
    ```
    docker start -i cub
    ```

# scPerb

Traditional methods for obtaining cellular responses after perturbation are usually labor-intensive and costly, especially when working with rare cells or under severe experimental conditions. Therefore, accurate prediction of cellular responses to perturbations is of great importance in computational biology. To address this problem, some methodologies have been previously developed, including graph-based approaches, vector arithmetic, and neural networks. However, these methods either mix the perturbation-related variances with the cell-type-specific pat- terns or implicitly distinguish them within black-box models. In this work, we introduce a novel framework, scPerb, to explicitly extract the perturbation-related variances and transfer them from perturbed data to control data. scPerb adopts the style transfer strategy by incorporating a style encoder into the architecture of a variational autoencoder. Such style encoder accounts for the differences in the latent representations between control cells and perturbed cells, which allows scPerb to accurately predict the gene expression data of perturbed cells. Through the comparisons with existing methods, scPerb presents improved performance and higher accuracy in predicting cellular responses to perturbations. Specifically, scPerb not only outperforms other methods across multiple datasets, but also achieves superior R2 values of 0.98, 0.98, and 0.96 on three benchmarking datasets.

## Highlights

- scPerb merged the concept of style transfer and VAE, resulting in incredible accuracy.
- scPerb outperforms all the existing models of single-cell perturbation prediction
- scPerb is developed and tailored for single-cell perturbation prediction and provided as a ready-to-use open-source software, demonstrating high accuracy and robust performance over existing methods.

## Link:

[yau-awards.com/uploads/file/20231031/20231031150434_30639.pdf](http://yau-awards.com/uploads/file/20231031/20231031150434_30639.pdf)

## FAQ

- **How can I install scPerb?**
    You can download scPerb from our GitHub link:

    ```
    git clone https://github.com/tongtongtot/scperb.git
    ```

    scPerb is built based on PyTorch, tested in Ubuntu 18.04, CUDA environment(cuda 11.2) the requirement packages include:

    ```
    torchvision==0.11.1
    torch==1.6.0
    tqdm==4.47.0
    numpy==1.13.3
    pandas==1.5.1
    ```

    or you can also use the following scripts:

    ```
    pip install -r requirements.txt
    ```

- **I want to try the toy demo, can I run scPerb in one command line?**
    You can use the following commands:

    ```
    python3 scperb.py
    ```

    or please refer to our training tutorial [here]$\color{red}{to\space be\space continued}$

- **Do I need a GPU for running scPerb?**
    scPerb can run on a standard laptop without GPU. For computational efficiency, we recommend you use a GPU. scPerb could detect whether there is an available GPU or not, so do not worry about this.

- **Can I generate my configuration file using the command line?**
    To enable users to generate their specific configure.yml with different hyperparameters, we provide a script to generate the configuration file. Please refer to [here]$\color{red}{to\space be\space continued}$ for details.

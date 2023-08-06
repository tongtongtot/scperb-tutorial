# scPerb

Inspired by the transfer learning paradigm, we presented scPerb, a generative model that can learn the "content" $Z_c^{ctrl}$ and $Z_c^{stim}$ of the cell types from both the control and stimulus datasets, where $c$ represented the "content features" of the cell types, and transfer the style $Z_s^{ctrl}$  from the control dataset to the stimulation dataset $Z_s^{stim}$ , where $s$ represented the "dataset styles".

scPerb is inspired by Varational Auto-Encoder (VAE) and the style-transfer GAN (stGAN). We used the variational inference to estimate the distribution $\mu$ and $\sigma$ of the "content features" $c$ in the latent space, and  project style input vector $r$ into the latent space and learn the transformation $s$ from the control dataset $X^{ctrl}$ to the stimulation dataset $X^{stim}$.  For the rest of the descriptions, we denote $E_\theta^c(.) $ as a content encoder to learn the cell-type awareness features, $E_\phi^s(.)$ to project the random-style input vectors to the latent space, $E_\mu^{c}(.)$ and $E_{\sigma}^{c}(.)$ represented to the $\mu$ and $\sigma$ estimation for the distribution of $c$, and $D_\Phi(.)$ for the decoder to generate the stimulation data from the latent variables $c$ and $s$. 

In the inference stage, given a specific cell type from the control dataset $X^{ctrl}$, scPerb will extract the cell type-related features $Z_c^{ctrl}$, and get the generated the pseudo-stimulus cell type $\hat{X}^{stim}$ based on $Z_c^{ctrl}$ and $\delta_s$, a result of a neuro-network, learning the difference between $Z_s^{ctrl}$ and $Z_s^{stim}$ . 

1. #### Encoders

We assumed the observations $X^{ctrl}$ and $X^{stim}$ from the control and stimulation datasets had two independent latent features: a cell type-related latent feature, denoted as  "content" $c$ , and a dataset-specific feature, denoted as "style" $s$. To extract the common cell type content feature, we first project the inputs into the latent space, then estimate the $\mu,\sigma$ to represent the normal distribution of $c$, and resample the latent variable $Z$ based on the generated distribution:
$$
\mu = E_\mu^{c}(E_\theta^c(X^{ctrl}))
$$

$$
\sigma = E_\sigma^{c}(E_\theta^c(X^{ctrl}))
$$

$$
Z_c^{ctrl} \sim N(\mu, \sigma)
$$

We shared the projection weights between the two dataset $X^{ctrl}$ and $X^{stim}$ , and therefore we can have the latent representation of $Z_c^{stim}$ as:
$$
\mu = E_\mu^{c}(E_\theta^c(X^{stim}))
$$

$$
\sigma = E_\sigma^{c}(E_\theta^c(X^{stim}))
$$

$$
Z_c^{stim} \sim N(\mu, \sigma)
$$

In this manuscript, our task is to generate the pseudo-stimulus cell types from the same cell types in the control dataset. Therefore, instead of learning the dataset styles explicitly, we applied a light-wise network to learn the transformation $s$ in the latent space. Our idea was inspired by the style transfer learnings \cite[StyleGAN], where randomly sampled a noise $r$ and project the latent space as the styles. In ScPerb, we applied a style encoder $E_\phi^s(.)$, which can project the $r$ into the latent space as the transformation variable to convert $Z^{ctrl}_c$ to $Z^{stim}_c$:
$$
s = E^s_ \phi(r)
$$

$$
\hat{Z}_c^{stim} = Z_c^{ctrl} + s
$$

And therefore we have the following style loss and the KL regulations:
$$
Style Loss = SmoothL1Loss(Z_c^{stim}, Z_c^{ctrl} + s)
$$

$$
KL Loss = KL(N(\mu, \sigma), N(0, I))
$$

Where SmoothL1Loss and KL divergence are:
$$
SmoothL1loss(x, y) = \left\{
\begin{align}
0.5 (x-y)^2 / \beta & \text{ if } |x-y| <\beta \\
|x - y| - 0.5\beta  & \text{  otherwise}
\end{align}
\right. 
$$

$$
KL (P, Q) = \sum_{x\in X} P(x) log(\frac{P(x)}{Q(x)})
$$



2. #### Decoder

We applied a decoder to generate the observations from the latent variables $\hat{Z}_c^{stim}$. Accordingly, the generated samples were denoted as: 
$$
\begin{gather}
\hat{X}^{stim} = D_\Phi(\hat{Z}_c^{stim})\\
\end{gather}
$$
Note that our task was to perturb the cell types from the control dataset to the stimulus dataset, instead of generating the samples from $Z_c^{stim}$ and $Z_c^{ctrl}$, we use $\hat{Z}_c^{stim}$. Therefore, our Generated Loss is:
$$
GenLoss = SmoothL1loss(x^{stim}, D_\Phi(Z_c^{ctrl} + \delta_s)
$$

3. #### Loss function

The objective functions will be combined with the Generated loss, Style Loss, and the KL regulation terms. 
$$
Loss = w_1 StyleLoss + w_2 KLLoss + w_3 GenLoss
$$


# Datasets and preprocess

Mohammad et al. [\cite{scgen}] included three groups of control and stimulated cells: two groups of PBMC cells, and a group of HPOLY cells. Mohammad et al. preprocessed the data by removing megakaryocytic cells, filtering the cells with a minimum of 500 expressing cells, extracting the top 6998 cells, and log-transforming the original data. All the data are available on https://github.com/theislab/scgen-reproducibility.

In our model, we performed further data preprocessing to ensure consistency between control and stimulus cells within each cell type. Specifically, for each cell type, we randomly selected an equal number of cells from both the control and stimulated groups and used them to balance the dataset. This data preprocessing step helped us create a more robust and unbiased dataset, enabling accurate and fair comparisons between each cell type's control and stimulus conditions during subsequent analyses. By doing such data processing, we guarantee that each pair of $X_{ctrl}$ and $X_{stim}$ have the same cell type, so the following style transfer process would be valid. 



#  Statistics and Reproduction

In ScPerb, we evaluated the performance of our model under a fixed seed of 42 by using the square of the $r_-value$, which is calculated by the $scipy.stats.linregress$ function. This metric measures the correlation between the generated images and the ground truth data. We computed the $r_-square$ values for all genes' mean and variance and the top 100 Differentially Expressed Genes (DEGs).

To understand the model’s results visually, we created scatter plots comparing the generated images to the corresponding ground truth data. This graph allowed us to observe how well the model's predictions aligned with the actual values.

Additionally, we investigated the differences between the generated images and the ground truth data for the top DEG using a violin plot. The DEGs were identified using the $scanpy.tl.rank_-genes_-groups$ function, employing the Wilcoxon method.

Through these analyses, we aimed to assess the accuracy and performance of our ScPerb model in generating realistic images based on the input gene expression data. The evaluation of $r_-square$ values and the visualization of the scatter and violin plots provided valuable insights into the model's capabilities and highlighted any discrepancies between the generated and true data for further investigation.



# Results

Fig 1

Fig 2

Fig 3

Fig 4

Fig 5



# Discussion

1. Summarize: 

    This is summarize.

2. Pros:

    This is the pros of our model.

3. Cons:

​		This is the cons of our model.

4. Future Improvement:

    In the future...

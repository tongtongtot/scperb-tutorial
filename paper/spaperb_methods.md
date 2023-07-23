# SpaPerb

Inspired by the transfer learning paradigm, we presented SpaPerb, a generative model that can learn the "content" $Z_c^{ctrl}$ and $Z_c^{stim}$ of the cell types from both the control and stimulus datasets, and transfer the style $Z_s^{ctrl}$  from the control dataset to the stimulus dataset $Z_s^{stim}$. In the inference stage, given a specific cell type from the control dataset $X^{ctrl}$, and a randomly generated variable $r$, SpaPerb will extract the cell type-related features $Z_c^{ctrl}$, and project $r$ to latent space as the dataset-specific transfer latent variable $\delta_s$ , and reconstruct the pseudo-stimulus cell type $\hat{X}^{stim}$ based on $Z_c^{ctrl}$ and $\delta_s$. We implement SpaPerb with the variational auto-encoder (VAE), with a content encoder $E_\theta^c(.) $ for cell-type awareness features and a style encoder $E_\phi^s(.)$ for the dataset transformation, and a decoder  $D_\Phi(.)$   to reconstruct the samples from the latent variables.

1. Encoders

We assumed the observations $X^{ctrl}$ and $X^{stim}$ from the control and stimulus datasets had two independent latent features: a cell type-related latent feature, denoted as  "content" $c$ , and a dataset-specific feature, denoted as "style" $s$. To extract the common cell type content feature, we applied a shared weight  encoder $E_\theta^c(.)$ to extract the features:
$$
\begin{gather}
& Z^{ctrl}_c = E_\theta^c(X^{ctrl})\\
& Z^{stim}_c = E_\theta^c(X^{stim})\\
\end{gather}
$$
In this manuscript, our task is to generate the pseudo-stimulus cell types from the same cell types in the control dataset. Therefore, instead of learning the dataset styles explicitly, we applied a light-wise network to learn the transformation $\delta_s$ in the latent space. Our idea was inspired by the style transfer learnings \cite[StyleGAN], where randomly sampled a noise $r$ and project the latent space as the styles. In SpaPerb, we applied a style encoder $E_\phi^s(.)$, which can project the $r$ into the latent space as the transformation variable to convert $Z^{ctrl}_c$ to $Z^{stim}_c$:
$$
\begin{gather}\\
\delta _s = E^s_ \phi(r) \\
\hat{Z}_c^{stim} = Z_c^{ctrl} + \delta _s
\end{gather}
$$

2. Decoder

We applied a decoder to generate the observations from the latent variables $\hat{Z}_c^{stim}$. Accordingly, the reconstruction samples were denoted as: 
$$
\begin{gather}
\hat{X}^{stim} = D_\Phi(\hat{Z}_c^{stim})\\
\end{gather}
$$
Note that our task was to perturb the cell types from the control dataset to the stimulus dataset, and therefore we didn't reconstruct the samples from $Z_c^{stim}$ and $Z_c^{ctrl}$. 



3. Variational Inference 

Based on our assumption, we have:
$$
P(x) = \int p(x|c; \theta)p(c) dc
$$
In the implementation, we assume $p(c)$ follows the standard Gaussian distribution $p(c)\sim N(0, I)$ , and a linear transformation between $Z_c^{ctrl}$ and $Z_c^{stim}$ in the latent space. Therefore, we have:
$$
P(x) = \int p(x | c; \theta) p(c) dc
$$
We use the Bayesian formula to calculate the posterior distribution $p(c|x)$ and  used the variational inference to maximize the ELBO with the Kullback-Leibler divergence:
$$
\begin{gather}
&KL(q_\theta(c|x) || p(c|x)) \\&= KL(q_\theta(c|x)||p(c)) \\&- E_{c\sim q_\theta(c|x)}[ln P(x|c)] + ln p(x)
\end{gather}
$$
 Where we can have 
$$
\begin{gather}
&ELBO = E_{c\sim q_\theta(c|x)}[ln P(x|c)]\\ &- KL(q_\theta(c|x) || p(c|x))\\ &\leq ln(p(x)) - KL(q_\theta(c|x)||p(c))
\end{gather}
$$
In our implementation, we used a linear layer to generate the $mu$ and another linear layer to generate the $variance$:
$$
\begin{gather}
& \mu = f_\Theta^{\mu}(z_c^{ctrl}) \\
& \sigma = f_\Theta^{\sigma}(z_c^{ctrl}) \\
\end{gather}
$$


4. Loss function

The objective functions will be combined with the reconstruction loss and the KL regulation terms. Additionally, we optimized the parameters in the style encoder by giving a constraint in the latent space:
$$
\begin{gather}
& Recon Loss = SmoothL1loss(x^{stim}, D_\Phi(Z_c^{ctrl} + \delta_s)\\
& KL Loss = KL(N(\mu, \sigma), N(0, I))\\
& Style Loss = SmoothL1Loss(Z_c^{stim}, Z_c^{ctrl} + \delta_s)\\
& Loss = w_1 * ReconLoss + w_2 * KLLoss + w_3 * StyleLoss
\end{gather}
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

# Datasets and preprocess

Mohammad et al. [\cite{scgen}] included three groups of control and stimulated cells: two groups of PBMC cells, and a group of HPOLY cells. Mohammad et al. preprocessed the data by removing megakaryocytic cells, filtering the cells with a minimum of 500 expressing cells, extracting the top 6998 cells, and log-transforming the original data. All the data are available on https://github.com/theislab/scgen-reproducibility.

In our model, we performed further data preprocessing to ensure consistency between control and stimulus cells within each cell type. Specifically, for each cell type, we randomly selected an equal number of cells from both the control and stimulated groups and used them to balance the dataset. This data preprocessing step helped us create a more robust and unbiased dataset, enabling accurate and fair comparisons between the control and stimulus conditions within each cell type during subsequent analyses. By doing such data processing, we guarantee that each pair of $X_{ctrl}$ and $X_{stim}$ have the same cell type, so the following style transfer process would be valid. 

#  Results and statistics

In SpaPerb, we evaluated the performance of our model using the square of the $r_-value$, which is calculated by the $scipy.stats.linregress$ function. This metric measures the correlation between the generated images and the ground truth data. We computed the $r_-square$ values for both the mean and variance of all genes, as well as for the top 100 Differentially Expressed Genes (DEGs).

To gain a visual understanding of the model's results, we created scatter plots comparing the generated images to the corresponding ground truth data. This graph allowed us to observe how well the model's predictions aligned with the actual values.

Additionally, we investigated the differences between the generated images and the ground truth data for the top DEG using a violin plot. The DEGs were identified using the $scanpy.tl.rank_-genes_-groups$ function, employing the Wilcoxon method.

Through these analyses, we aimed to assess the accuracy and performance of our SpaPerb model in generating realistic images based on the input gene expression data. The evaluation of $r_-square$ values and the visualization of the scatter and violin plots provided valuable insights into the model's capabilities and highlighted any discrepancies between the generated and true data for further investigation.

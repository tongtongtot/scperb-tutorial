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

![image-20230807215714466](/Users/tongtongtot/Library/Application Support/typora-user-images/image-20230807215714466.png)

![image-20230807215753116](/Users/tongtongtot/Library/Application Support/typora-user-images/image-20230807215753116.png)

##### scPerb is an innovative generative model that can accurately predict single-cell perturbation responses. 

The gene expressions from High-dimensional scRNA-seq are highly correlated with the single-cell responses. Traditionally, the algorithms focused on extracting the principle components from the gene expressions, along with the graph-based constraints\cite[scgen 23, 24]. Such methods were limited by the poor representation of the limited principle components, which only explore a limited portion of the raw data. More recently, neural networks have shown their potential in projecting the observations from raw data into the high-dimensional manifold, where the latent representations can approximate the real distribution beyond the raw data observations. The neural networks were data-driven methods, and can dynamically assign weights to the dominating features from the raw data. In particular, Variational Auto-Encoder (VAE) \cite[] assumed the raw observation can be sampled from a multivariate normal distribution, and used an encoder to estimate the mean and variance of the distribution, resample the latent variables and used the variational inference to generate new data observations from the estimated distribution. Conditional Variational Auto-Encoder (CVAE) \cite[] introduced more constraints to the neural network, allowing the end-users to generate more desired reconstructed samples for customized demands. In particular, scGEN\cite[] assumed a fixed linear gap between the control cells and the stimulation cells, calculated the latent difference \delta from both datasets, and predicted the stimulation cell response using latent representation from control cells and the calculated \delta. Meanwhile, Generative adversarial network (GAN) \cite[] introduced a creative but more black-box style in generating the data. GAN trained a “generator” to produce realistic “fake” data samples, while adversarially trained a discriminator to differentiate the “fake” data samples from the “real” data observations. The ultimate goal of GAN was to have a “generator” which can “fool” the discriminators when the loss converged. sc-WGAN \cite[] introduced GAN to the scRNA-seq data and obtain a universal representation manifold for different cell types. sc-WGAN can predict the single-cell perturbation response from the well-trained latent representations. Other researchers \cite[scgen] also introduced the style-transfer GAN (st-GAN), which aimed to transfer the dataset-related styles from the control dataset to the stimulation dataset. 

In our manuscript, we presented a novel tool named “scPerb” (shown Fig.1), which was inspired by the CVAE and style-transfer GAN. Suppose we have two datasets with different “styles” but the same cell types. We denoted $x_i^{ctrl}$ to represent the ith cell from the control dataset, and $x_j^{stim}$ for the jth cell from the stimulation dataset. scPerb decouples the perturbation task mentioned above into two steps: estimate the latent features of the cell types and a learnable dataset-related style transformation matrix. Inspired by the VAE architectures, scPerb first estimates the multi-variance normal distribution of the latent cell type feature c. Inspired by the style-transfer GAN,  scPerb uses a neural network to learn the style transformation matrix from the dataset. Compared with scGEN, which used a fixed vector $\delta$ to transfer the latent features from the control cells to the stimulation cells, scPerb introduced more learnable parameters and allowed the neural network dynamically assign the weights of the “style-transfer” vector based on the data. Therefore, scPerb can better learn both the style and content difference between the control and stimulation datasets, and output a better prediction compared to scGEN \cite[scGEN].

To demonstrate the performance of scPerb, we applied it to three datasets. Among these three datasets, two of them are two groups of published human peripheral blood mononuclear cells (PBMC) datasets stimulated with interferon ($IFN-\beta$) methods, and the rest is a group of intestinal epithelial cells fetched by parasitic helminth H.poly cells. We fairly compared our proposed scPerb with other benchmarking papers \cite[scgen, sc-WGAN, CVAE, st-GAN]. In this process, we first run all the cell types of one dataset in each model and combine the results for further processing. Then, we compare the prediction of all the methods with the ground truth, and the stimulation cells in the dataset, and get a final $R^2$ score.  Compared to all the other methods including scGEN, CVAE, style-transfer GAN and sc-WGAN, the predictions of scPerb are the most correlated with the cell types in the stimulation dataset. In the published human peripheral blood mononuclear cells (PBMC) dataset, scPerb gain a mean $R^2$ value of 0.98, while scGEN and CVAE only achieved 0.96 and 0.91 respectively. Moreover, both GAN-based methods style-transfer GAN and sc-WGAN  poorly predicted the perturbation response, resulting in $R^2$ values of 0.02 and 0.12 accordingly. In conclusion, scPerb best correlates the stimulation cells among all the other benchmarking methods. 

##### scPerb outperforms other benchmarkers

In the Study dataset (Fig2 (a)), we measure the mean $R^2$ among all cell types. Each $R^2$ reflected the specific correlation score between the prediction and the real stimulation cell type. scPerb achieved a mean $R^2$ of 0.98, which is higher than the second best benchmarking scGEN ($R^2=0.94$) and the third best bechmarking cVAE ($R^2=0.93$). Surprisingly, the GAN based methods had much worse performance. The stGAN and sc-WGAN only have $R^2=0.14$ and $R^2=0.10$ accordingly. When compared the perfomance with a specific cell type $CD4T$, scPerb achieved $R^2$=0.99, followed by scGEN and CVAE with $R^2=0.96$ and $R^2=0.95$.  stGAN and sc-wGAN had poor performance, with $R^2=0.01$ and $R^2=0.09$. 

Moreover, we inspected the correlation of a specific gene expression among the benchmarking papers. $FTL$ is **{the relation between FTL and CD4T, need refers}**. In our dataset, the gene expression of FTL in the control dataset is similar to the stimulation dataset. In this case, our scPerb focused on the mean of the gene expression in the real dataset and had a few high expressions. In contrast, scGEN, CVAE were more similar to the gene expression in the control dataset, while st-GAN and sc-WGAN had relatively smaller gene expression.  To support our observation, we also inspect another gene expression $IFIT2$ in the same $CD4T$ cell type. The gene expression of $IFIT2$ from control dataset are  full of zero values, while the gene expressions of $IFIT2$ in the stimulation dataset had higher values. In this case, scPerb, scgen and sc-WGAN can reflect the dataset difference, while CVAE and stGAN failed to transfer the gene expression from the control dataset to the stimulation dataset. Furthermore, we applied a wilcoxon test to examine whether the prediction and the real stimulation gene expression had significant difference. The pvalues showed that scPerb had 0.1763 in the prediction of $FTL$ and 0.0742 in the prediction of $IFIT2$, which failed to reject the hypothesis that the prediction and the real stimulation gene expression had signficiantly difference. Meanwhile, in the rest of the benchmarkings, the pvalues of scGEN was $6.2987e^{-15}$and $3.3282e^{-3}$, CVAE (pvalue=0.0307 for FTL and pvalue<0.0001. For the GAN-base methods, both the pvalues were smaller than 0.0001. Consequently, the pvalues from the benchmarking papers reject the hypothesis and indicating a significant difference between the prediction and real stimulation gene expressions. 

#### scPerb can accurately predict the perturbation of cells

We then explored the performance of scPerb in larger range of genes in more cell types. Figure3 (a) showed the scatter plot of the mean $R^2$ of all genes and the top 100 DEG genes. The mean of the prediction gene expression achieved $R^2=0.9905$ among all genes, and $R^2=0.9935$ among the top 100 DEG genes. In particular, we found that top 5 genes were $IFIT1, IFIT3, IF16, ISG20, ISG15$. 

Figure3 (b) explored the $R^2$ among difference cell types. The mean $R^2$ in B cells among all the genes and top 100 DEGs were $0.9708$, $0.9819$. For the rest of cell types, the $CD4T$, ..., $CD8T$ ,..., $CD14+Mono$, ..., $Dendritic$, ... , $FCGR3A+Mono$, ..., $NK$, ... 

$....$









##### scPerb is robust in various cell expression conditions

The resilience of scPerb was evidenced in scenarios where the mean expression of control differed from the stimulated type. The dot plot in Figure 3 demonstrates whether the mean expression of control cells is less (such as in genes IFIT1, IFIT2, and IFIT3), equal (such as in genes RPL3 and RPL13A), or greater than (such as in genes FTL, and ACTB) the perturbed cells, scPerb’s prediction remained close to the ground truth.

Fig 1

Fig 2

Fig 3

Fig 4

Fig 5



# Discussion

1. Summarize: 

    This is a summarize of the paper.

2. Pros:

    This is the pros of our model.

3. Cons:

​		This is the cons of our model.

4. Future Improvement:

    In the future...

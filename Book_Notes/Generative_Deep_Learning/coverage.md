# Generative Deep Learning: Coverage Map

Source: David Foster, Second edition, 2023. PDF page = printed page + 28.

Every parsed contents entry maps to a note below. This verifies navigational topic coverage, not sentence-level completeness.

| Chapter | Source topic | Printed page | Notes section |
|---|---|---|---|
| 1 | What Is Generative Modeling? | 4 | [Generative versus discriminative modeling](chapter-01-generative-modeling.html#topic-1) |
| 1 | Generative Versus Discriminative Modeling | 5 | [Generative versus discriminative modeling](chapter-01-generative-modeling.html#topic-1) |
| 1 | The Rise of Generative Modeling | 6 | [Generative versus discriminative modeling](chapter-01-generative-modeling.html#topic-1) |
| 1 | Generative Modeling and AI | 8 | [Generative versus discriminative modeling](chapter-01-generative-modeling.html#topic-1) |
| 1 | Our First Generative Model | 9 | [The first generative model and its assumptions](chapter-01-generative-modeling.html#topic-2) |
| 1 | Hello World! | 9 | [The first generative model and its assumptions](chapter-01-generative-modeling.html#topic-2) |
| 1 | The Generative Modeling Framework | 10 | [The first generative model and its assumptions](chapter-01-generative-modeling.html#topic-2) |
| 1 | Representation Learning | 12 | [Representation learning and latent spaces](chapter-01-generative-modeling.html#topic-3) |
| 1 | Core Probability Theory | 15 | [Core probability theory and maximum likelihood](chapter-01-generative-modeling.html#topic-4) |
| 1 | Generative Model Taxonomy | 18 | [Taxonomy of generative models](chapter-01-generative-modeling.html#topic-5) |
| 1 | The Generative Deep Learning Codebase | 20 | [Codebase, Docker, GPU, and reproducibility](chapter-01-generative-modeling.html#topic-6) |
| 1 | Cloning the Repository | 20 | [Codebase, Docker, GPU, and reproducibility](chapter-01-generative-modeling.html#topic-6) |
| 1 | Using Docker | 21 | [Codebase, Docker, GPU, and reproducibility](chapter-01-generative-modeling.html#topic-6) |
| 1 | Running on a GPU | 21 | [Codebase, Docker, GPU, and reproducibility](chapter-01-generative-modeling.html#topic-6) |
| 1 | Summary | 21 | [Codebase, Docker, GPU, and reproducibility](chapter-01-generative-modeling.html#topic-6) |
| 2 | Data for Deep Learning | 24 | [Data and high-level feature learning](chapter-02-deep-learning.html#topic-1) |
| 2 | Deep Neural Networks | 25 | [Data and high-level feature learning](chapter-02-deep-learning.html#topic-1) |
| 2 | What Is a Neural Network? | 25 | [Data and high-level feature learning](chapter-02-deep-learning.html#topic-1) |
| 2 | Learning High-Level Features | 26 | [Data and high-level feature learning](chapter-02-deep-learning.html#topic-1) |
| 2 | TensorFlow and Keras | 27 | [Data and high-level feature learning](chapter-02-deep-learning.html#topic-1) |
| 2 | Multilayer Perceptron (MLP) | 28 | [MLP project: preparing CIFAR-style image data](chapter-02-deep-learning.html#topic-2) |
| 2 | Preparing the Data | 28 | [MLP project: preparing CIFAR-style image data](chapter-02-deep-learning.html#topic-2) |
| 2 | Building the Model | 30 | [Building, compiling, and training an MLP](chapter-02-deep-learning.html#topic-3) |
| 2 | Compiling the Model | 35 | [Building, compiling, and training an MLP](chapter-02-deep-learning.html#topic-3) |
| 2 | Training the Model | 37 | [Building, compiling, and training an MLP](chapter-02-deep-learning.html#topic-3) |
| 2 | Evaluating the Model | 38 | [Evaluating the model](chapter-02-deep-learning.html#topic-4) |
| 2 | Convolutional Neural Network (CNN) | 40 | [Evaluating the model](chapter-02-deep-learning.html#topic-4) |
| 2 | Convolutional Layers | 41 | [Convolutional layers and tensor geometry](chapter-02-deep-learning.html#topic-5) |
| 2 | Batch Normalization | 46 | [Batch normalization and dropout](chapter-02-deep-learning.html#topic-6) |
| 2 | Dropout | 49 | [Batch normalization and dropout](chapter-02-deep-learning.html#topic-6) |
| 2 | Building the CNN | 51 | [Building, training, and evaluating the CNN](chapter-02-deep-learning.html#topic-7) |
| 2 | Training and Evaluating the CNN | 53 | [Building, training, and evaluating the CNN](chapter-02-deep-learning.html#topic-7) |
| 2 | Summary | 54 | [Building, training, and evaluating the CNN](chapter-02-deep-learning.html#topic-7) |
| 3 | Introduction | 60 | [Autoencoders and the Fashion-MNIST project](chapter-03-variational-autoencoders.html#topic-1) |
| 3 | Autoencoders | 61 | [Autoencoders and the Fashion-MNIST project](chapter-03-variational-autoencoders.html#topic-1) |
| 3 | The Fashion-MNIST Dataset | 62 | [Autoencoders and the Fashion-MNIST project](chapter-03-variational-autoencoders.html#topic-1) |
| 3 | The Autoencoder Architecture | 63 | [Autoencoders and the Fashion-MNIST project](chapter-03-variational-autoencoders.html#topic-1) |
| 3 | The Encoder | 64 | [Encoder, decoder, and reconstruction](chapter-03-variational-autoencoders.html#topic-2) |
| 3 | The Decoder | 65 | [Encoder, decoder, and reconstruction](chapter-03-variational-autoencoders.html#topic-2) |
| 3 | Joining the Encoder to the Decoder | 67 | [Encoder, decoder, and reconstruction](chapter-03-variational-autoencoders.html#topic-2) |
| 3 | Reconstructing Images | 69 | [Encoder, decoder, and reconstruction](chapter-03-variational-autoencoders.html#topic-2) |
| 3 | Visualizing the Latent Space | 70 | [Why plain autoencoders are awkward generators](chapter-03-variational-autoencoders.html#topic-3) |
| 3 | Generating New Images | 71 | [Why plain autoencoders are awkward generators](chapter-03-variational-autoencoders.html#topic-3) |
| 3 | Variational Autoencoders | 74 | [Why plain autoencoders are awkward generators](chapter-03-variational-autoencoders.html#topic-3) |
| 3 | The Encoder | 75 | [VAE encoder and the reparameterization trick](chapter-03-variational-autoencoders.html#topic-4) |
| 3 | The Loss Function | 80 | [Loss function: reconstruction and KL divergence](chapter-03-variational-autoencoders.html#topic-5) |
| 3 | Training the Variational Autoencoder | 82 | [Loss function: reconstruction and KL divergence](chapter-03-variational-autoencoders.html#topic-5) |
| 3 | Analysis of the Variational Autoencoder | 84 | [Training and analyzing the latent space](chapter-03-variational-autoencoders.html#topic-6) |
| 3 | Exploring the Latent Space | 85 | [Training and analyzing the latent space](chapter-03-variational-autoencoders.html#topic-6) |
| 3 | The CelebA Dataset | 85 | [Training and analyzing the latent space](chapter-03-variational-autoencoders.html#topic-6) |
| 3 | Training the Variational Autoencoder | 87 | [Training and analyzing the latent space](chapter-03-variational-autoencoders.html#topic-6) |
| 3 | Analysis of the Variational Autoencoder | 89 | [Training and analyzing the latent space](chapter-03-variational-autoencoders.html#topic-6) |
| 3 | Generating New Faces | 90 | [New faces, latent arithmetic, and morphing](chapter-03-variational-autoencoders.html#topic-7) |
| 3 | Latent Space Arithmetic | 91 | [New faces, latent arithmetic, and morphing](chapter-03-variational-autoencoders.html#topic-7) |
| 3 | Morphing Between Faces | 92 | [New faces, latent arithmetic, and morphing](chapter-03-variational-autoencoders.html#topic-7) |
| 3 | Summary | 93 | [New faces, latent arithmetic, and morphing](chapter-03-variational-autoencoders.html#topic-7) |
| 4 | Introduction | 96 | [GAN and DCGAN architecture](chapter-04-generative-adversarial-networks.html#topic-1) |
| 4 | Deep Convolutional GAN (DCGAN) | 97 | [GAN and DCGAN architecture](chapter-04-generative-adversarial-networks.html#topic-1) |
| 4 | The Bricks Dataset | 98 | [GAN and DCGAN architecture](chapter-04-generative-adversarial-networks.html#topic-1) |
| 4 | The Discriminator | 99 | [The discriminator and generator](chapter-04-generative-adversarial-networks.html#topic-2) |
| 4 | The Generator | 101 | [The discriminator and generator](chapter-04-generative-adversarial-networks.html#topic-2) |
| 4 | Training the DCGAN | 104 | [Training the DCGAN](chapter-04-generative-adversarial-networks.html#topic-3) |
| 4 | Analysis of the DCGAN | 109 | [Training the DCGAN](chapter-04-generative-adversarial-networks.html#topic-3) |
| 4 | GAN Training: Tips and Tricks | 110 | [Analysis and training tips](chapter-04-generative-adversarial-networks.html#topic-4) |
| 4 | Wasserstein GAN with Gradient Penalty (WGAN-GP) | 113 | [Analysis and training tips](chapter-04-generative-adversarial-networks.html#topic-4) |
| 4 | Wasserstein Loss | 114 | [Wasserstein loss and Lipschitz constraints](chapter-04-generative-adversarial-networks.html#topic-5) |
| 4 | The Lipschitz Constraint | 115 | [Wasserstein loss and Lipschitz constraints](chapter-04-generative-adversarial-networks.html#topic-5) |
| 4 | Enforcing the Lipschitz Constraint | 116 | [Wasserstein loss and Lipschitz constraints](chapter-04-generative-adversarial-networks.html#topic-5) |
| 4 | The Gradient Penalty Loss | 117 | [Gradient penalty and WGAN-GP training](chapter-04-generative-adversarial-networks.html#topic-6) |
| 4 | Training the WGAN-GP | 119 | [Gradient penalty and WGAN-GP training](chapter-04-generative-adversarial-networks.html#topic-6) |
| 4 | Analysis of the WGAN-GP | 121 | [Gradient penalty and WGAN-GP training](chapter-04-generative-adversarial-networks.html#topic-6) |
| 4 | Conditional GAN (CGAN) | 122 | [Conditional GANs](chapter-04-generative-adversarial-networks.html#topic-7) |
| 4 | CGAN Architecture | 123 | [Conditional GANs](chapter-04-generative-adversarial-networks.html#topic-7) |
| 4 | Training the CGAN | 124 | [Conditional GANs](chapter-04-generative-adversarial-networks.html#topic-7) |
| 4 | Analysis of the CGAN | 126 | [Conditional GANs](chapter-04-generative-adversarial-networks.html#topic-7) |
| 4 | Summary | 127 | [Conditional GANs](chapter-04-generative-adversarial-networks.html#topic-7) |
| 5 | Introduction | 130 | [Autoregressive factorization](chapter-05-autoregressive-models.html#topic-1) |
| 5 | Long Short-Term Memory Network (LSTM) | 131 | [Autoregressive factorization](chapter-05-autoregressive-models.html#topic-1) |
| 5 | The Recipes Dataset | 132 | [Recipe data, tokenization, and training windows](chapter-05-autoregressive-models.html#topic-2) |
| 5 | Working with Text Data | 133 | [Recipe data, tokenization, and training windows](chapter-05-autoregressive-models.html#topic-2) |
| 5 | Tokenization | 134 | [Recipe data, tokenization, and training windows](chapter-05-autoregressive-models.html#topic-2) |
| 5 | Creating the Training Set | 137 | [Recipe data, tokenization, and training windows](chapter-05-autoregressive-models.html#topic-2) |
| 5 | The LSTM Architecture | 138 | [Embedding layer and LSTM architecture](chapter-05-autoregressive-models.html#topic-3) |
| 5 | The Embedding Layer | 138 | [Embedding layer and LSTM architecture](chapter-05-autoregressive-models.html#topic-3) |
| 5 | The LSTM Layer | 140 | [Embedding layer and LSTM architecture](chapter-05-autoregressive-models.html#topic-3) |
| 5 | The LSTM Cell | 142 | [Embedding layer and LSTM architecture](chapter-05-autoregressive-models.html#topic-3) |
| 5 | Training the LSTM | 144 | [Training and sampling the LSTM](chapter-05-autoregressive-models.html#topic-4) |
| 5 | Analysis of the LSTM | 146 | [Training and sampling the LSTM](chapter-05-autoregressive-models.html#topic-4) |
| 5 | Recurrent Neural Network (RNN) Extensions | 149 | [Stacked RNNs, GRUs, and bidirectional cells](chapter-05-autoregressive-models.html#topic-5) |
| 5 | Stacked Recurrent Networks | 149 | [Stacked RNNs, GRUs, and bidirectional cells](chapter-05-autoregressive-models.html#topic-5) |
| 5 | Gated Recurrent Units | 151 | [Stacked RNNs, GRUs, and bidirectional cells](chapter-05-autoregressive-models.html#topic-5) |
| 5 | Bidirectional Cells | 153 | [Stacked RNNs, GRUs, and bidirectional cells](chapter-05-autoregressive-models.html#topic-5) |
| 5 | PixelCNN | 153 | [Stacked RNNs, GRUs, and bidirectional cells](chapter-05-autoregressive-models.html#topic-5) |
| 5 | Masked Convolutional Layers | 154 | [PixelCNN: masked convolutions and residual blocks](chapter-05-autoregressive-models.html#topic-6) |
| 5 | Residual Blocks | 156 | [PixelCNN: masked convolutions and residual blocks](chapter-05-autoregressive-models.html#topic-6) |
| 5 | Training the PixelCNN | 158 | [PixelCNN: masked convolutions and residual blocks](chapter-05-autoregressive-models.html#topic-6) |
| 5 | Analysis of the PixelCNN | 159 | [PixelCNN analysis and mixture distributions](chapter-05-autoregressive-models.html#topic-7) |
| 5 | Mixture Distributions | 162 | [PixelCNN analysis and mixture distributions](chapter-05-autoregressive-models.html#topic-7) |
| 5 | Summary | 164 | [PixelCNN analysis and mixture distributions](chapter-05-autoregressive-models.html#topic-7) |
| 6 | Introduction | 168 | [Normalizing flows and change of variables](chapter-06-normalizing-flows.html#topic-1) |
| 6 | Normalizing Flows | 169 | [Normalizing flows and change of variables](chapter-06-normalizing-flows.html#topic-1) |
| 6 | Change of Variables | 170 | [Normalizing flows and change of variables](chapter-06-normalizing-flows.html#topic-1) |
| 6 | The Jacobian Determinant | 172 | [Jacobian determinant and composition](chapter-06-normalizing-flows.html#topic-2) |
| 6 | The Change of Variables Equation | 173 | [Jacobian determinant and composition](chapter-06-normalizing-flows.html#topic-2) |
| 6 | RealNVP | 174 | [RealNVP and the two-moons dataset](chapter-06-normalizing-flows.html#topic-3) |
| 6 | The Two Moons Dataset | 174 | [RealNVP and the two-moons dataset](chapter-06-normalizing-flows.html#topic-3) |
| 6 | Coupling Layers | 175 | [RealNVP and the two-moons dataset](chapter-06-normalizing-flows.html#topic-3) |
| 6 | Training the RealNVP Model | 181 | [Training, density analysis, and sampling](chapter-06-normalizing-flows.html#topic-4) |
| 6 | Analysis of the RealNVP Model | 184 | [Training, density analysis, and sampling](chapter-06-normalizing-flows.html#topic-4) |
| 6 | Other Normalizing Flow Models | 186 | [Glow, FFJORD, and architectural trade-offs](chapter-06-normalizing-flows.html#topic-5) |
| 6 | GLOW | 186 | [Glow, FFJORD, and architectural trade-offs](chapter-06-normalizing-flows.html#topic-5) |
| 6 | FFJORD | 187 | [Glow, FFJORD, and architectural trade-offs](chapter-06-normalizing-flows.html#topic-5) |
| 6 | Summary | 188 | [Glow, FFJORD, and architectural trade-offs](chapter-06-normalizing-flows.html#topic-5) |
| 7 | Introduction | 189 | [Energy and the Boltzmann distribution](chapter-07-energy-based-models.html#topic-1) |
| 7 | Energy-Based Models | 191 | [Energy and the Boltzmann distribution](chapter-07-energy-based-models.html#topic-1) |
| 7 | The MNIST Dataset | 192 | [MNIST and the energy function](chapter-07-energy-based-models.html#topic-2) |
| 7 | The Energy Function | 193 | [MNIST and the energy function](chapter-07-energy-based-models.html#topic-2) |
| 7 | Sampling Using Langevin Dynamics | 194 | [Langevin dynamics sampling](chapter-07-energy-based-models.html#topic-3) |
| 7 | Training with Contrastive Divergence | 197 | [Contrastive divergence and replay](chapter-07-energy-based-models.html#topic-4) |
| 7 | Analysis of the Energy-Based Model | 201 | [Analysis, limitations, and other energy models](chapter-07-energy-based-models.html#topic-5) |
| 7 | Other Energy-Based Models | 202 | [Analysis, limitations, and other energy models](chapter-07-energy-based-models.html#topic-5) |
| 7 | Summary | 203 | [Analysis, limitations, and other energy models](chapter-07-energy-based-models.html#topic-5) |
| 8 | Introduction | 206 | [Denoising models, score models, and flower data](chapter-08-diffusion-models.html#topic-1) |
| 8 | Denoising Diffusion Models (DDM) | 208 | [Denoising models, score models, and flower data](chapter-08-diffusion-models.html#topic-1) |
| 8 | The Flowers Dataset | 208 | [Denoising models, score models, and flower data](chapter-08-diffusion-models.html#topic-1) |
| 8 | The Forward Diffusion Process | 209 | [Forward diffusion and reparameterization](chapter-08-diffusion-models.html#topic-2) |
| 8 | The Reparameterization Trick | 210 | [Forward diffusion and reparameterization](chapter-08-diffusion-models.html#topic-2) |
| 8 | Diffusion Schedules | 211 | [Diffusion schedules](chapter-08-diffusion-models.html#topic-3) |
| 8 | The Reverse Diffusion Process | 214 | [Reverse diffusion and noise prediction](chapter-08-diffusion-models.html#topic-4) |
| 8 | The U-Net Denoising Model | 217 | [The U-Net denoiser](chapter-08-diffusion-models.html#topic-5) |
| 8 | Training the Diffusion Model | 224 | [Training, EMA, and reverse sampling](chapter-08-diffusion-models.html#topic-6) |
| 8 | Sampling from the Denoising Diffusion Model | 225 | [Training, EMA, and reverse sampling](chapter-08-diffusion-models.html#topic-6) |
| 8 | Analysis of the Diffusion Model | 228 | [Analysis and sampling trade-offs](chapter-08-diffusion-models.html#topic-7) |
| 8 | Summary | 231 | [Analysis and sampling trade-offs](chapter-08-diffusion-models.html#topic-7) |
| 9 | Introduction | 236 | [GPT and the wine-review dataset](chapter-09-transformers.html#topic-1) |
| 9 | GPT | 236 | [GPT and the wine-review dataset](chapter-09-transformers.html#topic-1) |
| 9 | The Wine Reviews Dataset | 237 | [GPT and the wine-review dataset](chapter-09-transformers.html#topic-1) |
| 9 | Attention | 238 | [Attention, queries, keys, and values](chapter-09-transformers.html#topic-2) |
| 9 | Queries, Keys, and Values | 239 | [Attention, queries, keys, and values](chapter-09-transformers.html#topic-2) |
| 9 | Multihead Attention | 241 | [Multihead attention and causal masking](chapter-09-transformers.html#topic-3) |
| 9 | Causal Masking | 242 | [Multihead attention and causal masking](chapter-09-transformers.html#topic-3) |
| 9 | The Transformer Block | 245 | [Transformer blocks and positional encoding](chapter-09-transformers.html#topic-4) |
| 9 | Positional Encoding | 248 | [Transformer blocks and positional encoding](chapter-09-transformers.html#topic-4) |
| 9 | Training GPT | 250 | [Training and analyzing GPT](chapter-09-transformers.html#topic-5) |
| 9 | Analysis of GPT | 252 | [Training and analyzing GPT](chapter-09-transformers.html#topic-5) |
| 9 | Other Transformers | 255 | [Other transformers: T5, GPT-3/GPT-4, and ChatGPT](chapter-09-transformers.html#topic-6) |
| 9 | T5 | 256 | [Other transformers: T5, GPT-3/GPT-4, and ChatGPT](chapter-09-transformers.html#topic-6) |
| 9 | GPT-3 and GPT-4 | 259 | [Other transformers: T5, GPT-3/GPT-4, and ChatGPT](chapter-09-transformers.html#topic-6) |
| 9 | ChatGPT | 260 | [Other transformers: T5, GPT-3/GPT-4, and ChatGPT](chapter-09-transformers.html#topic-6) |
| 9 | Summary | 264 | [Other transformers: T5, GPT-3/GPT-4, and ChatGPT](chapter-09-transformers.html#topic-6) |
| 10 | Introduction | 268 | [ProGAN and progressive training](chapter-10-advanced-gans.html#topic-1) |
| 10 | ProGAN | 269 | [ProGAN and progressive training](chapter-10-advanced-gans.html#topic-1) |
| 10 | Progressive Training | 269 | [ProGAN and progressive training](chapter-10-advanced-gans.html#topic-1) |
| 10 | Outputs | 276 | [ProGAN and progressive training](chapter-10-advanced-gans.html#topic-1) |
| 10 | StyleGAN | 277 | [StyleGAN mapping and synthesis networks](chapter-10-advanced-gans.html#topic-2) |
| 10 | The Mapping Network | 278 | [StyleGAN mapping and synthesis networks](chapter-10-advanced-gans.html#topic-2) |
| 10 | The Synthesis Network | 279 | [StyleGAN mapping and synthesis networks](chapter-10-advanced-gans.html#topic-2) |
| 10 | Outputs from StyleGAN | 280 | [StyleGAN mapping and synthesis networks](chapter-10-advanced-gans.html#topic-2) |
| 10 | StyleGAN2 | 281 | [StyleGAN2 modulation and demodulation](chapter-10-advanced-gans.html#topic-3) |
| 10 | Weight Modulation and Demodulation | 282 | [StyleGAN2 modulation and demodulation](chapter-10-advanced-gans.html#topic-3) |
| 10 | Path Length Regularization | 283 | [Path-length regularization and no progressive growing](chapter-10-advanced-gans.html#topic-4) |
| 10 | No Progressive Growing | 284 | [Path-length regularization and no progressive growing](chapter-10-advanced-gans.html#topic-4) |
| 10 | Outputs from StyleGAN2 | 286 | [Path-length regularization and no progressive growing](chapter-10-advanced-gans.html#topic-4) |
| 10 | Other Important GANs | 286 | [Path-length regularization and no progressive growing](chapter-10-advanced-gans.html#topic-4) |
| 10 | Self-Attention GAN (SAGAN) | 286 | [Path-length regularization and no progressive growing](chapter-10-advanced-gans.html#topic-4) |
| 10 | BigGAN | 288 | [Self-attention GAN and BigGAN](chapter-10-advanced-gans.html#topic-5) |
| 10 | VQ-GAN | 289 | [VQ-GAN: discrete latent image tokens](chapter-10-advanced-gans.html#topic-6) |
| 10 | ViT VQ-GAN | 292 | [ViT VQ-GAN and synthesis of ideas](chapter-10-advanced-gans.html#topic-7) |
| 10 | Summary | 294 | [ViT VQ-GAN and synthesis of ideas](chapter-10-advanced-gans.html#topic-7) |
| 11 | Introduction | 298 | [Music as structured data and Bach cello suites](chapter-11-music-generation.html#topic-1) |
| 11 | Transformers for Music Generation | 299 | [Music as structured data and Bach cello suites](chapter-11-music-generation.html#topic-1) |
| 11 | The Bach Cello Suite Dataset | 300 | [Music as structured data and Bach cello suites](chapter-11-music-generation.html#topic-1) |
| 11 | Parsing MIDI Files | 300 | [Music as structured data and Bach cello suites](chapter-11-music-generation.html#topic-1) |
| 11 | Tokenization | 303 | [Parsing MIDI, tokenization, and training sets](chapter-11-music-generation.html#topic-2) |
| 11 | Creating the Training Set | 304 | [Parsing MIDI, tokenization, and training sets](chapter-11-music-generation.html#topic-2) |
| 11 | Sine Position Encoding | 305 | [Sine positional encoding and multiple streams](chapter-11-music-generation.html#topic-3) |
| 11 | Multiple Inputs and Outputs | 307 | [Sine positional encoding and multiple streams](chapter-11-music-generation.html#topic-3) |
| 11 | Analysis of the Music-Generating Transformer | 309 | [Analysis of the music-generating transformer](chapter-11-music-generation.html#topic-4) |
| 11 | Tokenization of Polyphonic Music | 313 | [Polyphonic grid versus event tokenization](chapter-11-music-generation.html#topic-5) |
| 11 | MuseGAN | 317 | [MuseGAN data and generator](chapter-11-music-generation.html#topic-6) |
| 11 | The Bach Chorale Dataset | 317 | [MuseGAN data and generator](chapter-11-music-generation.html#topic-6) |
| 11 | The MuseGAN Generator | 320 | [MuseGAN data and generator](chapter-11-music-generation.html#topic-6) |
| 11 | The MuseGAN Critic | 326 | [MuseGAN critic, analysis, and limitations](chapter-11-music-generation.html#topic-7) |
| 11 | Analysis of the MuseGAN | 327 | [MuseGAN critic, analysis, and limitations](chapter-11-music-generation.html#topic-7) |
| 11 | Summary | 329 | [MuseGAN critic, analysis, and limitations](chapter-11-music-generation.html#topic-7) |
| 12 | Introduction | 331 | [Reinforcement learning and CarRacing](chapter-12-world-models.html#topic-1) |
| 12 | Reinforcement Learning | 332 | [Reinforcement learning and CarRacing](chapter-12-world-models.html#topic-1) |
| 12 | The CarRacing Environment | 334 | [Reinforcement learning and CarRacing](chapter-12-world-models.html#topic-1) |
| 12 | World Model Overview | 336 | [Architecture and staged training](chapter-12-world-models.html#topic-2) |
| 12 | Architecture | 336 | [Architecture and staged training](chapter-12-world-models.html#topic-2) |
| 12 | Training | 338 | [Architecture and staged training](chapter-12-world-models.html#topic-2) |
| 12 | Collecting Random Rollout Data | 339 | [Architecture and staged training](chapter-12-world-models.html#topic-2) |
| 12 | Training the VAE | 340 | [Training and exploring the VAE](chapter-12-world-models.html#topic-3) |
| 12 | The VAE Architecture | 341 | [Training and exploring the VAE](chapter-12-world-models.html#topic-3) |
| 12 | Exploring the VAE | 343 | [Training and exploring the VAE](chapter-12-world-models.html#topic-3) |
| 12 | Collecting Data to Train the MDN-RNN | 346 | [MDN-RNN data, architecture, and sampling](chapter-12-world-models.html#topic-4) |
| 12 | Training the MDN-RNN | 346 | [MDN-RNN data, architecture, and sampling](chapter-12-world-models.html#topic-4) |
| 12 | The MDN-RNN Architecture | 347 | [MDN-RNN data, architecture, and sampling](chapter-12-world-models.html#topic-4) |
| 12 | Sampling from the MDN-RNN | 348 | [MDN-RNN data, architecture, and sampling](chapter-12-world-models.html#topic-4) |
| 12 | Training the Controller | 348 | [MDN-RNN data, architecture, and sampling](chapter-12-world-models.html#topic-4) |
| 12 | The Controller Architecture | 349 | [Controller and CMA-ES](chapter-12-world-models.html#topic-5) |
| 12 | CMA-ES | 349 | [Controller and CMA-ES](chapter-12-world-models.html#topic-5) |
| 12 | Parallelizing CMA-ES | 351 | [Controller and CMA-ES](chapter-12-world-models.html#topic-5) |
| 12 | In-Dream Training | 353 | [In-dream training and simulator exploitation](chapter-12-world-models.html#topic-6) |
| 12 | Summary | 356 | [In-dream training and simulator exploitation](chapter-12-world-models.html#topic-6) |
| 13 | Introduction | 360 | [DALL-E 2 architecture and text-image alignment](chapter-13-multimodal-models.html#topic-1) |
| 13 | DALL.E 2 | 361 | [DALL-E 2 architecture and text-image alignment](chapter-13-multimodal-models.html#topic-1) |
| 13 | Architecture | 362 | [DALL-E 2 architecture and text-image alignment](chapter-13-multimodal-models.html#topic-1) |
| 13 | The Text Encoder | 362 | [DALL-E 2 architecture and text-image alignment](chapter-13-multimodal-models.html#topic-1) |
| 13 | CLIP | 362 | [DALL-E 2 architecture and text-image alignment](chapter-13-multimodal-models.html#topic-1) |
| 13 | The Prior | 367 | [The prior, decoder, and DALL-E examples](chapter-13-multimodal-models.html#topic-3) |
| 13 | The Decoder | 369 | [The prior, decoder, and DALL-E examples](chapter-13-multimodal-models.html#topic-3) |
| 13 | Examples from DALL.E 2 | 373 | [The prior, decoder, and DALL-E examples](chapter-13-multimodal-models.html#topic-3) |
| 13 | Imagen | 377 | [Imagen and DrawBench](chapter-13-multimodal-models.html#topic-4) |
| 13 | Architecture | 377 | [Imagen and DrawBench](chapter-13-multimodal-models.html#topic-4) |
| 13 | DrawBench | 378 | [Imagen and DrawBench](chapter-13-multimodal-models.html#topic-4) |
| 13 | Examples from Imagen | 379 | [Imagen and DrawBench](chapter-13-multimodal-models.html#topic-4) |
| 13 | Stable Diffusion | 380 | [Stable Diffusion and latent diffusion](chapter-13-multimodal-models.html#topic-5) |
| 13 | Architecture | 380 | [Stable Diffusion and latent diffusion](chapter-13-multimodal-models.html#topic-5) |
| 13 | Examples from Stable Diffusion | 381 | [Stable Diffusion and latent diffusion](chapter-13-multimodal-models.html#topic-5) |
| 13 | Flamingo | 381 | [Stable Diffusion and latent diffusion](chapter-13-multimodal-models.html#topic-5) |
| 13 | Architecture | 382 | [Flamingo: vision encoder and Perceiver Resampler](chapter-13-multimodal-models.html#topic-6) |
| 13 | The Vision Encoder | 382 | [Flamingo: vision encoder and Perceiver Resampler](chapter-13-multimodal-models.html#topic-6) |
| 13 | The Perceiver Resampler | 383 | [Flamingo: vision encoder and Perceiver Resampler](chapter-13-multimodal-models.html#topic-6) |
| 13 | The Language Model | 385 | [Flamingo language model, examples, and comparison](chapter-13-multimodal-models.html#topic-7) |
| 13 | Examples from Flamingo | 388 | [Flamingo language model, examples, and comparison](chapter-13-multimodal-models.html#topic-7) |
| 13 | Summary | 389 | [Flamingo language model, examples, and comparison](chapter-13-multimodal-models.html#topic-7) |
| 14 | Timeline of Generative AI | 392 | [Timeline: VAE/GAN, transformer, and big-model eras](chapter-14-conclusion.html#topic-1) |
| 14 | 2014�2017: The VAE and GAN Era | 394 | [Timeline: VAE/GAN, transformer, and big-model eras](chapter-14-conclusion.html#topic-1) |
| 14 | 2018�2019: The Transformer Era | 394 | [Timeline: VAE/GAN, transformer, and big-model eras](chapter-14-conclusion.html#topic-1) |
| 14 | 2020�2022: The Big Model Era | 395 | [Timeline: VAE/GAN, transformer, and big-model eras](chapter-14-conclusion.html#topic-1) |
| 14 | The Current State of Generative AI | 396 | [Large language models and text-to-code](chapter-14-conclusion.html#topic-2) |
| 14 | Large Language Models | 396 | [Large language models and text-to-code](chapter-14-conclusion.html#topic-2) |
| 14 | Text-to-Code Models | 400 | [Large language models and text-to-code](chapter-14-conclusion.html#topic-2) |
| 14 | Text-to-Image Models | 402 | [Text-to-image and other applications](chapter-14-conclusion.html#topic-3) |
| 14 | Other Applications | 405 | [Text-to-image and other applications](chapter-14-conclusion.html#topic-3) |
| 14 | The Future of Generative AI | 407 | [Everyday life, work, and education](chapter-14-conclusion.html#topic-4) |
| 14 | Generative AI in Everyday Life | 407 | [Everyday life, work, and education](chapter-14-conclusion.html#topic-4) |
| 14 | Generative AI in the Workplace | 409 | [Everyday life, work, and education](chapter-14-conclusion.html#topic-4) |
| 14 | Generative AI in Education | 410 | [Everyday life, work, and education](chapter-14-conclusion.html#topic-4) |
| 14 | Generative AI Ethics and Challenges | 411 | [Ethics and challenges](chapter-14-conclusion.html#topic-5) |
| 14 | Final Thoughts | 413 | [Final thoughts: active inference and continued learning](chapter-14-conclusion.html#topic-6) |

Preface/foreword provide background; index, reference lists, and full exercise/code listings remain in the source book.

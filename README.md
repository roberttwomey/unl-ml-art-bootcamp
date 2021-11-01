![machine generated images](ml_art2019.png)
# Machine Learning for the Arts Bootcamp

Johnny Carson Center for Emerging Media Arts | 5:30-8:00pm, November 10, 2021 | [Center for Emerging Media Arts](https://maps.unl.edu/CEMA)

Prof. Robert Twomey | rtwomey@unl.edu | [roberttwomey.com](http://roberttwomey.com)

Register here: [https://forms.gle/pL8aWC8JJuKitBKFA](https://forms.gle/pL8aWC8JJuKitBKFA)

## Description

This bootcamp is a short introduction to Art and Machine Learning (ML), in advance of the [EMAR349 Machine Learning for the Arts](http://ml.roberttwomey.com) course in Spring 2022. The goal is to introduce both the technical software tools and the conceptual application domain of ML and Art. Students who thrive in this bootcamp workshop will enjoy the semester-long course.

With recent advancements in machine learning techniques, researchers have demonstrated remarkable achievements in image synthesis (BigGAN, StyleGAN), text generation (GPT-3), and other areas of generative and perceptive ML. This hands-on workshop introduces state-of-the-art techniques for text-to-image translation, where textual prompts are used to guide the generation of visual imagery. Participants will gain experience with Open AI's CLIP network and Google's BigGAN, using jupyter notebooks in either google Colab or HCC Open On Deman, which they can apply to their own work after the event. We will discuss other relationships between text and image in art and literature; consider the strengths and limitations of these new techniques; and relate these computational processes to human language, perception, and visual expression and imagination. __Please bring a text you would like to experiment with!__

No skills are necessary for the workshop, although some experience coding (EMAR161 or equivalent introductory coding experience with Javascript or Python) will be helpful.

## Schedule
|    Time    | Activity |
|------------|----|
| 5:30	| Introductions; Open up jupyter; Introduction to Neural Nets, Generative Adversarial Networks (GANs), Generative Text (Transformers). |
| 5:45  | Hands on with jupyter notebook and image generation: BigGAN; talk about latent vectors and GAN generation; talk about interpolations. Explore outputs. |
| 6:15	| Check in on results: Particpants share BigGAN results as a group; Q & A | 
| 6:30  | Hands on with text-to-image translation: CLIP + BigGAN + CMA-ES; Talk about format of textual "prompts"/inputs; Explore visual outputs. |
| 7:00	| Check in on results: Participants informally share work with group; Q&A about challenges/techniques. Participants continue working. |
| 7:15	| Hands on with interplolation videos: Interpolation and latent walks. |
| 7:30	| Full group Discussion, Future Directions | 
| 8:00  | End |

## Notebooks

Click on the links below to open the corresponding notebooks in google colab (will be replaced with HCC). You can only run one at a time.

1. BigGAN - [BigGAN_handson.ipynb](https://colab.research.google.com/github/roberttwomey/machine-imagination-workshop/blob/main/BigGAN_handson.ipynb)
2. Text to Image Generation with BigGAN and CLIP - [text_to_image_BiGGAN_CLIP.ipynb](https://colab.research.google.com/github/roberttwomey/machine-imagination-workshop/blob/main/text_to_image_BigGAN_CLIP.ipynb)
3. Generate latent interpolations - [generate_from_stored.ipynb](https://colab.research.google.com/github/roberttwomey/machine-imagination-workshop/blob/main/generate_from_stored.ipynb)
4. Batch process textual prompts - text_to_image_batch.ipynb (not yet implemented on colab)

## Discussion

- How do words specify/suggest/evoke images? 
- What do you see when you read? Are some texts more or less imagistic?
- How can we use this artificial machine imagination to understand our human visual imagination? 
- How might you incorporate these techniques into our creative production or scholarship? 
- What would it mean to diversify machine imagination?

## References
- Google Deep Mind BigGAN, [Large Scale GAN Training for High Fidelity Natural Image Synthesis](https://deepmind.com/research/publications/large-scale-gan-training-high-fidelity-natural-image-synthesis), 2018
  - see the BigGAN hands-on notebook above to get a sense for image generation with BigGAN, noise vectors, truncation, and latent interpolation. 
- NVIDIA StyleGAN2, [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948), 2019
  - see for example [https://thispersondoesnotexist.com/](https://thispersondoesnotexist.com/), a photorealistic face generator with StyleGAN2
- OpenAI GPT-3: [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165), 2020
  - see Kenric Allado-McDowell's [Pharmako-AI](https://ignota.org/products/pharmako-ai) for an example a book written with GPT-3.
- OpenAI [CLIP: Connecting Text and Image](https://openai.com/blog/clip/), 2021
- OpenAI [DALL-E: Creating Images from Text](https://openai.com/blog/dall-e/), 2021
  - the interactive examples on this page will give you a sense of the kind of technique we will explore during the workshop.
- Good [list of CLIP-related to text-to-image notebooks on Reddit r/MachineLearning](https://www.reddit.com/r/MachineLearning/comments/ldc6oc/p_list_of_sitesprogramsprojects_that_use_openais/)

## Networks

__Neural Network__

<!-- ![image](https://user-images.githubusercontent.com/1598545/118535440-55f57f00-b6ff-11eb-8a67-9282039abc19.png)-->
<!-- <img width="600" alt="simple feed forward network" src="https://user-images.githubusercontent.com/1598545/118535440-55f57f00-b6ff-11eb-8a67-9282039abc19.png"> -->

<!-- ![image](https://user-images.githubusercontent.com/1598545/118535575-7b828880-b6ff-11eb-9fd8-40486111e3d8.png)-->
<img width="600" alt="mnist digit classifier network" src="https://user-images.githubusercontent.com/1598545/118535575-7b828880-b6ff-11eb-9fd8-40486111e3d8.png">

Neural Networks, or Artificial Neural Networks (ANNs) are networks (graphs) composed of nodes and edges, loosely modelled on the architecture of biological brain. They are generally composed of distinct layers of neurons, where outputs from one feed inputs of another. Broadly, each node resembles a neuron, accepting inputs from a number of other nodes, and defined with its own activiation function, bias, and forward connections. There are many variations on this basic architecture. Above we see a very simple fully connected, feed forward network that takes as an input 28 x 28 pixel grayscale images (784 input signals), and produces a 0-10 digit classifier on the output. Neural networks are used for many generative and predictive tasks across sound, image, text, etc.

__Generative Adversarial Networks (GANs)__

<!--![image](https://user-images.githubusercontent.com/1598545/118530742-d74a1300-b6f9-11eb-9743-6d87c96961a3.png)-->
<!-- cropped ![image](https://user-images.githubusercontent.com/1598545/118531573-d5348400-b6fa-11eb-8f53-a324929ef48c.png)-->
<img width="600" alt="GAN diagram with generator and discriminator" src="https://user-images.githubusercontent.com/1598545/118531573-d5348400-b6fa-11eb-8f53-a324929ef48c.png">

A Generative Adversarial Network (GAN) is a kind of generative model. The basic idea is to set up a game between two players (game theory). The Generator creates samples that resemble the input dataset. The Discriminator evaluates samples to determine if they are real or fake (binary classifier). We can think of the generator as being like a counterfeiter, trying to make fake money, and the discriminator as being like police, trying to allow legitimate money and catch counterfeit money. To succeed in this game, the counterfeiter must learn to make money that is indistinguishable from genuine money, and the generator network must learn to create samples that are drawn from the same distribution as the training data. (adversarial) Both networks are trained simultaneously.

Ian Goodfellow introduced the architecture in __Generative Adversarial Nets__, Goodfellow et. al (2014) https://arxiv.org/pdf/1406.2661.pdf

__BigGAN__

<!-- ![image](https://user-images.githubusercontent.com/1598545/118533146-8daef780-b6fc-11eb-8f4a-91b205fb65b5.png)-->
<img width="600" alt="samples from BigGAN" src="https://user-images.githubusercontent.com/1598545/118533146-8daef780-b6fc-11eb-8f4a-91b205fb65b5.png">

BigGAN (2018) set a standard for high resolution, high fidelity image synthesis in 2018. It contained four times as many parameters and eight times the batch size of previous models, and synthesized a state of the art 512 x 512 pixel image across [1000 different classes](https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt) from [Imagenet](https://www.image-net.org/). It was also prohibitively expensive to train! Thankfully Google/Google Brain has released a number of pretrained models for us to explore. Read the paper here https://arxiv.org/abs/1809.11096.

__CLIP__

<!--![image](https://user-images.githubusercontent.com/1598545/118530808-ee890080-b6f9-11eb-8a49-1e1e73097792.png)-->
<img width="600" alt="CLIP diagram" src="https://user-images.githubusercontent.com/1598545/118530808-ee890080-b6f9-11eb-8a49-1e1e73097792.png">

CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs. It can be instructed in natural language to predict the most relevant text snippet, given an image, without directly optimizing for the task, similarly to the zero-shot capabilities of GPT-2 and 3. We found CLIP matches the performance of the original ResNet50 on ImageNet “zero-shot” without using any of the original 1.28M labeled examples, overcoming several major challenges in computer vision. 

CLIP pre-trains an image encoder and a text encoder to predict which images were paired with which texts in our dataset. We then use this behavior to turn CLIP into a zero-shot classifier. We convert all of a dataset’s classes into captions such as “a photo of a dog” and predict the class of the caption CLIP estimates best pairs with a given image. 

CLIP learns from unfiltered, highly varied, and highly noisy data ... text–image pairs that are already publicly available on the internet. See details on the [CLIP Model Card](https://github.com/openai/CLIP/blob/main/model-card.md#data)

To learn more about CLIP, try the Interacting with CLIP colab: https://colab.research.google.com/github/openai/clip/blob/master/notebooks/Interacting_with_CLIP.ipynb)

(from https://github.com/openai/CLIP)


<!--

# == NOTE: EVERYTHING BELOW HERE WILL BE REVISED ==
**Topics**:
- Generative Methods in the Arts
- Neural Style Transfer
- MNIST Style Recognition
- Text Generation with RNNs
- GAN Latent Space Exploration

**Work Environment**:
- Kubernetes (datahub.ucsd.edu)
- Jupyterhub (cuda, python, etc. all pre-installed)
- TensorFlow
- Participant-contributed ideas + data (images, text, etc.)

## Schedule
| Time | Activity |
| --- | --- |
| 10:00-10:30 | Lecture: Introduction and Motivation ("Generative Machines") |
| 10:30-11:15 | Style transfer activity with online tools. Discussion |
| 11:15-11:30 | Hands on with Python and Tensorflow in Jupyterhub |
| 11:30-12:00 | Notebook 1: Style Transfer in Python |
| 12:00-12:30 | LUNCH BREAK |
| 12:45-1:00 | Generative vs. Perceptive ML |
| 1:00-1:30 | Notebook 2: Fashion MNIST |
| 1:30-2:00 | Notebook 3: Text Generation with LSTMs |
| 2:00-2:30 | Notebook 4: GAN Latent Space Exploration |
| 2:30-3:00 | Closing Discussion. Sign up for class |

## Schedule Description

**10:00-10:30 - Lecture** - For the first half hour the instructor will briefly survey generative methods in the arts, and describe some contemporary ML and generative tools. https://drive.google.com/file/d/1H62Sy2ETbOPl9Q1aYSZVU__RWJO8wB0h/view?usp=sharing

**10:30-11:00 - Style Transfer Activity** - The instructor will briefly introduce neural style transfer, and demonstrate online tools to accomplish style transfers. In a quick “sketching” exercise, students will have 20 minutes to experiment with style transfer software. They will select both the source style images and target images, and are asked to consider the meanings and aesthetics of their decisions, implementing one of the following:

1. Content and Style that are Incongruous
2. Content and Style that Amplify Each Other 
3. Style that adds an emotional charge to the content

**11:00-11:15 - Style Transfer Discussion** - Compare results in small groups, and choose one output to share with the class. 

**11:15-11:30 - Hands on with Python and Tensorflow in jupyterhub** - Instructor will introduce our ML software development environment. We will use the TensorFlow framework through python notebooks running in jupyterhub. For the bootcamp, these resources are pre-configured as virtual machine images running on Kubernetes, to skip time consuming software install processes. 

1. Log-on to [datahub.ucsd.edu](http://datahub.ucsd.edu)
2. Select `ECE188_SP19_A00: Scientific Python + Machine Learning Tools (1 GPU, 2 CPU, 16GB RAM)`. Spawn. 
3. Clone this repository into your account: On your jupyter Notebook Dashboard screen, open a terminal (**New->Terminal**) and run the following command at your prompt: ```git clone https://github.com/roberttwomey/ml-art-bootcamp```
4. You should be ready to go! Switch to: [0_Schedule.ipynb](0_Schedule.ipynb)

**11:30-12:00 - Notebook 1: Style Transfer in Python** - Students will work with a provided partial implementation of neural style transfer in TensorFlow, completing the implementation, and experimenting with their own style transfer function to produce new images. 

**12:00-12:30 - Lunch Break -**

**12:45-1:00 - Discuss Generative vs Perceptive ML**

**1:00-1:30 - Notebook 3: MNIST Activity** - Students will work through a provided MNIST Fashion recognition example in TensorFlow, training a recognition model, and then running it on photos of fashion samples they capture with their cellphones to explore the inferential capabilities and limitations of their model.

**1:30-2:00 - Notebook 3: Text Generation with RNNs** - 

**2:00-2:30 - Notebook 4: GAN Latent Space Exploration** - Instructor will introduce advanced generative methods for images, including Generative Adversarial Networks (GANs). Students will work with provided code to experiment with a GAN. 

**2:30-3:00 - Closing Discussion** - Students will briefly share their progress from the GAN activity and we will have a closing discussion. Instructor will answer questions about the course, and students will sign up for course if interested. 
-->

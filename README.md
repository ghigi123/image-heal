# image-heal
Ce projet a pour but de proposer une solution d'inpainting par deux approches de deeplearning.
Ces deux approches pemettent de générer une large partie manquante d'une image, en assurant la continuité de l'image sur la zone de transition et la cohérence de la sémantique de l'image.
Notre première approche est un DCGAN(deep convolutional generative adversarial networks) avec des coûts "contextuels", pour assurer la cohérence sémantique de l'image et "perceptuels" pour réaliser l'inpainting. La deuxième approche est basée sur les auto-encodeurs, avec une fonction de coût adversariale pour améliorer le niveau de détail des résultats


### Prerequisites
A installer avant de lancer le projet :
- pytorch
- torchvision

### Utilisation

Mettez votre jeu de donnée dans un dossier, les jeux de données que nous avons utilisés sont Labeled Faces in the Wild (LFW) et Common Objects in Context (COCO).

Le fichier main.py permet de lancer training et complétion des images

```
usage: main.py [-h] [--batch-size N] [--mode MODE] [--test-batch-size N]
               [--epochs N] [--output-dir OUTPUT_DIR] [--no-cuda] [--seed S]
               [--log-interval N] [--data-path DATA_PATH]
               [--image-size IMAGE_SIZE] [--method METHOD]
               [--discriminator-model-name DISCRIMINATOR_MODEL_NAME]
               [--generator-model-name GENERATOR_MODEL_NAME]
               [--mask-size MASK_SIZE]

Image inpainting with pytorch

optional arguments:
  -h, --help            show this help message and exit
  --batch-size N        input batch size for training (default: 64)
  --mode MODE           train or complete
  --test-batch-size N   input batch size for testing (default: 1000)
  --epochs N            number of epochs to train (default: 10)
  --output-dir OUTPUT_DIR
                        folder to output images and model checkpoints
  --no-cuda             disables CUDA training
  --seed S              random seed (default: 1)
  --log-interval N      how many batches to wait before logging training
                        status
  --data-path DATA_PATH
                        relative path to a folder containing a folder
                        containing images to learn from
  --image-size IMAGE_SIZE
                        image will be resized to this size
  --method METHOD       which method to use (context-encoder or dcgan)
  --discriminator-model-name DISCRIMINATOR_MODEL_NAME
  --generator-model-name GENERATOR_MODEL_NAME
  --mask-size MASK_SIZE
                        size of the mask used for inpainting
```

Exemple d'usage :

```
python main.py --data-path ./data/lfw-deepfunneled --mode train --output-dir encoder-lfw-128 --image-size 128 --method context-encoder --mask-size 60 --epochs 25
```

Les deux méthodes possibles sont: 'dcgan' et 'context-encoder', pour choisir entre les deux approches développées. Les deux modes sont : 'train' et 'complete', qui servent respectivement à entraîner les système pour un jeu de données et à compléter l'image endommagée.


### Test auto-encodeur
Le fichier `train_autoencoder.py` permet de tester l'entrainement de l'auto-encodeur en utilisant MSE comme unique fonction de coût.
On utilise `build_mask` pour définir les caractéristiques du masque (positionné au hasard ou centré ; dimension).
L'idée de ce fichier est essentiellement de tester différentes architectures de l'auto-encodeur (réduire les artefacts, comparer à la sortie etc ...).

```
python train_autoencoder.py
```

## Auteurs

* **Ghislain Janneau**
* **Arnault Chazareix**
* **Elodie Ikkache**

## Remerciements

Approche DCGAN :

* http://bamos.github.io/2016/08/09/deep-completion/

Approche context encoder :

* http://people.eecs.berkeley.edu/~pathak/context_encoder/

Approche vision :

* http://graphics.cs.cmu.edu/projects/scene-completion/

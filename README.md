# image-heal
Ce projet a pour but de proposer une solution d'inpainting par deux approches de deeplearning.
Ces deux approches pemettent de générer une large partie manquante d'une image, en assurant la continuité de l'image sur la zone de transition et la cohérence de la sémantique de l'image.
Notre première approche est un DCGAN(deep convolutional generative adversarial networks) avec des coûts "contextuels", pour assurer la cohérence sémantique de l'image et "perceptuels" pour réaliser l'inpainting. La deuxième approche est basée sur les auto-encodeurs, avec une fonction de coût adversariale pour améliorer le niveau de détail des résultats

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites
A installer avant de lancer le projet :
- pytorch
- torchvision

### Installing

???
Mettez votre jeu de donnée dans data, les jeux de données que nous avons utilisés sont Labeled Faces in the Wild (LFW) et Common Objects in Context (COCO).

Lancer main.py avec comme arguments method et mode.
Les deux méthodes possibles sont: '' et 'context-encoder', pour choisir entre les deux approches développées. Les deux modes sont : 'train' et 'complete', qui servent respectivement à entraîner les système pour un jeu de données et à compléter l'image endommagée.

## Deployment

???

## Built With
???

## Contributing
???

## Versioning

???

## Authors

* **Ghislain Janneau**
* **Arnault Chazareix**
* **Elodie Ikkache**

## License

???

## Acknowledgments
pour l'approche dcgan:
http://bamos.github.io/2016/08/09/deep-completion
pour l'approche auto-encodeur:
https://github.com/pathak22/context-encoder

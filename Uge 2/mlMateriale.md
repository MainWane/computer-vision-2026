## Uge 2 - Machine learning med PyTorch

Først installér PyTorch, et af de machine learning biblioteker der finder bredest anvendelse
* Vælg den version der passer bedst til din computer ud fra denne tabel: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) 

### Basics: Klargøring af data, træning af modeller, definition af modeller, viden om modeller

* Lave sit eget billededatasæt til PyTorch: [https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files)
* Definere lag i en neural model: [https://docs.pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html](https://docs.pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html)
* Eksempler fra basis, og bygger én vigtig machine learning detalje på én efter én [https://docs.pytorch.org/tutorials/beginner/pytorch_with_examples.html](https://docs.pytorch.org/tutorials/beginner/pytorch_with_examples.html)

### Forskellige modelarkitekturer

Efter vi har lært fra basics, at klargøre datasæt, at lave om på lagene i en model, og kunne forbedre en model en smule med parametre, så er det tid til at lære hvordan forskellige kendte modelarkitekturer er bygget op.

* YOLO. En af de mest simple arkitekturer der finder anvendelse til differentiering mellem flere klasser. Den er egentligt ret ringe, men meget let at implementere, og kræver ikke så mange resourcer hvilket gør det muligt ret billigt at implementere den. [https://www.geeksforgeeks.org/machine-learning/yolo-you-only-look-once-real-time-object-detection/](https://www.geeksforgeeks.org/machine-learning/yolo-you-only-look-once-real-time-object-detection/)

* VGG-Net. En slags "moderarkitektur", altså baserer mange mere avancerede/sofistikerede netværk sig på denne. Den er ikke meget anderledes end den vi bruger i det basale eksempel ovenover. [https://www.geeksforgeeks.org/computer-vision/vgg-net-architecture-explained/](https://www.geeksforgeeks.org/computer-vision/vgg-net-architecture-explained/)

* ResNet - større model, bedre til at differentiere mange forskellige objekter. Den er delvist baseret på VGG19. Bliv ikke skræmt af de mange lag, mønsteret er bare flere convolution lag med tiltagende større kernels. [https://www.geeksforgeeks.org/deep-learning/residual-networks-resnet-deep-learning/](https://www.geeksforgeeks.org/deep-learning/residual-networks-resnet-deep-learning/)

Optimering af træning

Beskrivelser af de parametre der er justerbare og påvirker træning: https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
    
Visualisér modeller og log træningsstatistikker: https://docs.pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html

Alle typer loss funktioner, layers osv. der er tilgængelige via PyTorch API'en: https://docs.pytorch.org/docs/stable/nn.html#loss-functions
    
Profiler: Check hardware performance for din model samt track antal kald: https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
    
Confusion matrix - En lidt bedre måde at vise modellens performance end accuracy: https://sklearn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

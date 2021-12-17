# DeBERTer: Interpretable Debate Argument Quality Identification

This GitHub repository contains all the code for project DeBERTer (paper link coming soon). This work was done as a final project for the class EECS 595: Natural Language Processing at the University of Michigan. The authors are Junliang Huang, Nameer Hirschkind, and Zewen Wu. Below are instructions on how to reproduce our results.

## Environment Setup
Please run the following command line arguments to set up the proper python virtual environment for this project.
```
python3 -m venv [your_env]
pip install -r requirement.txt
source [yout_env]/bin/activate
```
## LSTM Models

### Training
To train any LSTM model, you only need to run the `trainLSTM.py` file. All parameters for the model are specified in lines 7 to 17 of this file. The first time you run it, you will have to set `load_cache = False` to run preprocessing on the entire train dataset one time. On later runs, you can set `load_cache = True` and the program will load previously preprocessed data, speeding up the procedure. We recommend leaving the number of epochs, learning rate, and batch size intact to reproduce our results. The `use_attention` variable determines whether to run the Multiheaded attention LSTM variant or the Concatenation variant of the model, both of which are described in our paper. The `learned_embedings` variable simply determines whether or not gradients are passed to the embedding layer of the LSTM, which is always intitialized with GloVe word vectors by default. The `num_outputs` variable has no effect on model performance and can be ignored (it is only there to enable the addition of dummy variables to the output). Note that LSTMS train quite quickly on a laptop (usually 5-10 minutes for 20 epochs).

### Testing
To test an LSTM model, you need only run `testLSTM.py`. Simply set the `SAVEPATH` variable to the name of the model you trained, and it will automatically be loaded. When testing is complete, an accuracy score will be printed to the console.

## ELECTRA Models

### Training
To train a model with an ELECTRA backbone, you just have to run the poorly named file `trainBERT.py`. All of the model settings are determined from lines 7 to 11 of the file and are a subset of the settings described in the LSTM training section. One difference is that if you want to change the architecture variant you are using, you must actually modify the initialization of the backbone class on line 30. Simply replace what is there with the name of the desired architecture you want to train (the options are `DistilBERTDotProduct`, `DistilBERTAttention`, `DistilBERTSimple`, which are the Contrastive, Attention, and Concatenation variants described in the paper respectively).

### Testing
To test an ELECTRA model, run `testBERT.py` exactly as you would for an LSTM model test.

## LIME Interpretation
All the code to generate LIME interpretations is contained in the `lime-BERT.ipynb` and `lime-LSTM.ipynb` Jupyter notebooks.

## Attention Interpretation
Currently the code to generate attention weight heatmap is in another branch `interpret-attention` since the backbone code need to be modified. Please check out `attint-LSTM.ipynb`.

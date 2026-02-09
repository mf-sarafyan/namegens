# Namegens Exercise

This repo contains some study on NLP with neural networks using PyTorch. 

The idea was to build character-level models for fantasy name generation, inspired by Andrej Karpathy's awesome series on this (at https://www.youtube.com/@AndrejKarpathy). So there's a lot of implementation that already exists within torch or other libs, but is here for study and debugging reasons. 

The repo is structured in "parts" as ideas evolved and got more complex. I'm trying to record my journey and thoughts as the model evolves (and so does my understanding of the area). 

Most of the code here is either taken from Andrej's talks, or implemented together with AI agents. 

## Part 1 

Just a simple MLP implementation from Andrej's lectures.

## Part 2 

Includes a more torchified code with a way larger model, more data, and some generation. Also includes some debugging of the activations and gradients to ensure the model is training correctly and explore functionality.

## Part 3 

Part 3 went wrong! I kept it around to figure out what happened. 

This is a more complex multi-head attention model - based on Andrej's attention lecture - where I played around with the data generation quite a bit and probably screwed something up. This model learnt only to return the control character "." most of the time. I think there are some errors in the model implementation and in the data processing. 

It might have been an issue with the weight normalization as well. 

## Part 4 

This time attention worked; I built it up based on the previous mlp_torchified implementation. Interestingly the model validation scores are not improving with this, probably because the model is overfitting to the training data. This is not really my objective yet, so I'm leaving it as-is. 

## Part 5 

Part 5 is an evolution of part 4 where I'm taking the category data from the dataset (races for the Forgotten Realms dataset and category from the kagglehub one) and inputs it to the model as a query in a cross-attention layer (mostly because it sounded fun). Now the model can get a category input as a conditional for the name generation. The layer works by using the category embedding as the Query vector (and the sequence as key and value). 

It doesn't work very well - overfits like hell -, but right now I'm mostly playing around with the architecture, before commiting to an idea I like and trying to optimize the model.

## Part 6 

Beautified version - just organizing everything a bit better and more professionally to iterate easier. One major change already has been adding skip connections to all the layers. 

## Part 7

Part 7 started out with just adding byte-pair encoding tokenization but, since that seemed like the last major step to take, it evolved into a rework of the architecture as well. 

These are the steps I took, in order:
- Adding another cross-attention layer where the Query was the sequence and key, value were the category embedding. This is because I felt like the category signal was too weak and didn't affect the outputs as much as I expected. 
- Adding an Adaptive Layer Norm as the first layer that controlled the sequence inputs to the rest of the network based on the category. This is because I still felt like the category signal was too weak. This structure also doesn't get the skip connection, so it should be very effective at altering the outputs 
    - and it was; on generation I could see the names matched the categories a lot better. For categories with a lot of data. Categories were pretty sparse (some had like one or two samples, even).
    - A good verification that this worked well was that for the category "bears" it generated the name "bear". 
- Grouping the categories together. My friend ChatGPT helped me group the 500-something categories into about 35 groups. This eliminated the issue of categories with tiny samples and improved the results significantly.

And I think that is enough for this project. Whenever I need some NPC names I can come back to it!


# Getting started 
- have something that can run notebooks
- python -m venv .env 
- activate your venv (either `source .env\bin\activate` or `.env\Scripts\activate.bat` on windows)
- pip install requirements.txt


### Data Sources 

https://github.com/alxgiraud/fantasygen/blob/master/data/defaultDictionaries.json

https://archive.org/stream/story_games_name_project/the_story_games_names_project_djvu.txt

https://www.darkshire.net/jhkim/rpg/lordoftherings/names.pdf

data.data_scraping.py is a simple code that scrapes the contents of the DnD Forgotten Realms wiki for character names together with their race categories. llm_parse_names.py is an auxiliary code to parse the e-book story games names project into a categorized csv file.

```
import kagglehub

# Download latest version
path = kagglehub.dataset_download("isaacbenge/fantasy-for-markov-generator")

print("Path to dataset files:", path)
``` 

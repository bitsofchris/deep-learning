# Deep Learning in Public

This repo is a collection of resources, notes, and code as I learn in public to build time series Transformer models.

Like Richard Feynman believed - if you can't explain it simply or create it from scratch, you don't really know it.

If you'd like to learn more about me or follow my work head to [bitsofchris.com](https://bitsofchris.com).


# Project Milestones

## Simple Neural Network (In-Progress)
Goal: Implement a basic neural network from scratch to understand these foundational concepts.

Deliverable:
- [DONE](code/000_neural-network-xor/nn_SGD.py) A network that solves XOR with no ML libraries
- [DONE](code/001_nn-mnist-image-classification/nn-pytorch.py) A network that solves image classification (maybe with PyTorch b/c I want to learn PyTorch)
- Implementing advanced techniques from [chapter 3](http://neuralnetworksanddeeplearning.com/chap3.html) from scratch on one of my networks

## A basic Transformer Model
Goal: Implement the Transformer architecture to understand attention mechanisms.


Deliverable:
- I have Sebastian Raschka's book How to Code a LLM from Scratch to use here
- Not sure yet, but I'd like to try and tokenize candlesticks from price charts as my "language"

## A time series Transformer Model on a dataset I create
Goal: Thinking back to my day trading days, create a good dataset that can allow a Transformer to search for similar patterns in history and offer probabilities of a next move.

Deliverable:
- A dataset
- A model that is fine tuned for searching the history of financial data for similar scenarios

This will likely evolve as I work and understand what's possible. 

I strongly believe that how you represent the problem in the data is the most important piece in making a useful model. I'm biased because I was a trader first, then data engineer who is only now trying to catch up on the modeling side, but I still belive the data is most important.

# What is this Repo?

For more on this repo see this [daily-blog entry](daily-blog/2024-10-05%20-%20Data%20Engineer%20to%20timeseries%20Transformer%20models.md) which is copied below:

### TLDR
This is a daily log of my learning in public.

I will build a time series Transformer model using financial data from the ground up.

I am a former day trader, currently working as a data engineer, trying to learn the AI side of things.

### The Challenge
I am a Staff data engineer on an AI research team. As a former trader, I can't help but see potential applications of timeseries Transformer models in finance. (hold the eye roll please)

My work time will have me dive deeper into the data engineering work to prepare data for model training. Things like PyTorch, Ray, and Spark.

This repo is to log my path to understanding the AI research side of things. How do we design neural networks? How do we build custom Transformer models on time series data or any data? 

In short, this is my ultralearning project in public to become a deep learning expert.

My unfair advantages here are that I am currently working as a data engineer on a team building time series Transformer models with observability data. I've been a data engineer for over 7 years, after being a day trader and algorithmic trading developer for the preceeding 7 years.

Now it's time to build up my expertise in AI - but from the ground up, from the foundations.

My current project goal is to build my own Transformer models that tokenize financial time series data in various ways to experiment with. This is where my domain expertise as a former trader I think can really help - once I know what is possible on the modeling side, I've got a lot of things I want to play with.

Learning in public and through a project is a potent combination to learn things deeply.

Follow along as I post here 6 days per week about what I learn on the path to building a custom time series Transformer model.

When necessary I'll walk backwards to understand the basics. I am starting today with a simple neural network from scratch.

I won't share all my second brain notes but I'll share daily logs like this, any code I write, and the resources I use to learn. I may also just share my LLM conversations as I use them to help me learn.

### Reflection
I've tried a few versions of learning in public and have been looking for a way to force myself to create daily while learning towards a specific project goal.

A GitHub repo seems like a great passive way to work in public, have notes alongside my code, and allow people to follow along without me having to spam posts on LinkedIn or my Substack.

#### Why bother writing about it?
A few reasons: 
- I love externalizing my notes, it helps me retain what I learn. 
- I love the thinking ability that expands by trying to write on your own.
- By forcing myself to reflect and share what I learn daily - I'll be noticing things to learn more and helping myself learn it deeper by sharing.

My concerns are though that having a repo separate from Obsidian second brain will be an operational overhead distracting from my main goal of just learning this stuff well.

I'll write these daily posts in my second brain and then copy paste to the repo. I'll work this out over time.

I can do 6 days per week, maybe 7 if feeling up for it but I like to reserve Sundays as a free day to work on whatever or nothing.


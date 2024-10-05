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


### TIL
I am implementing a neural network from scratch to solve the XOR problem.

First goal is to get the feedforward process working.

Feedforward means given some inputs lets feed that through our network and produce an output.
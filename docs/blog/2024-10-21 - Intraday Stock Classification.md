# 2024-10-21 - Feature Engineering for stock project

Over the weekend I was inspired to try a neural network on some financial time series data (didn't take me very long to pivot here - lol).

So taking a break from the book I am working through to build a small project.

The Goal: A feed-forward neural network using hourly financial data to predict - is the next interval up or down?

This led to a long conversation with Claude and searching around with Perplexity to understand how I can/should do feature engineering for this problem. I've added some of those notes in today's PR.

And started today with just fetching the data using `yfinance`, I grabbed 2 years for three symbols for a total of about 10k training samples. I will likely add more symbols here to get this up to 100k samples perhaps but for now this is good enough.

My notebook looks good in creating features here.

### Next steps
- [ ] Normalize the data I've created - and get it ready for model training.
	- [ ] Get more data to make train, validation, eval sets?
- [ ] Build & Train the network

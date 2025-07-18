



0. Check that the data is probably useful

Looks like it as there's a range of valuations for each player

1. Define the Problem - Small

We're a Data Scientist for Chelsea.
The transfer strategy over the last few years of ownership has been to buy young players where they can be sold for a profit.

### Aims

Create a model to estimate players values over time. 

**Benefits**

* Determine players who's value may increase over the next few years that we could buy, then sell.

* Determine players who we own who's value may decrease that we should sell. 

* Determine players who we own who's value may increase that we should perhaps offer a new contract

* Look at what causes players value to increase / decrease.

**Caveats**

* This needs to be able to be explained to stakeholders
* Real value is what people would pay for so understanding real transfer fees would be good
    
2. Research Related Work - often forgotten - Small

* https://medium.com/@paudelsamir/machine-learning-for-football-player-market-value-prediction-end-to-end-project-b776ee25880b

* https://kritjunsree.medium.com/the-science-of-football-advanced-data-analysis-of-player-movements-via-transfermarkt-com-5efd9daf9107

3. Understand the Data - Crap in / Crap out - Large

* Understand how Value is determined

### Analytical Questions

#### Create documentation for this  

* What data can we get time horizons for 

* Do the most valuable players and their trajactories make sense to me

Yes - I've looked at Messi ,Vinni and Hazard and it makes a sense check to me
* How does age impact it

Looking only at the premier league, the average value follows roughly a normal distribution peaking at 25

* How does position impact it 

It does - the more attacking are the most valuable, by a large factor.
Goalkeepers are considerably less valuable.
This is true for all leagues and the premier league

* How does time length in the contract affect it 

There is a clear difference where for the Premier League it peaks at 5 Years, then drops off
 
* Does this vary by league

* Are individual players skewing the averages

* How much is inflation / year affecting it

For the Premier League From 2005 to 2017 it's relatively static, but this is probably because as the years have gone on more players have been added to the dataset.
The players that wouldn't have been in it previously are more likely to be for lower ranking teams

* How has the dataset changed over the years / how much data is there 

There's gradually more data over the years

* How does this correlate with actual transfer cost 

Transfer cost is on average below the market value, but when people whos value is 0 or transfer cost was 0 are removed then it's relatively equal.
Intrestingly there's already transfers in for 2026, which should be filtered out 

* Is there data for the release clause 

No 

* How does transfer value change based on contract time length 

4. Plan for Validation

Tracking RMSE would be a good start - this isn't good for non normalised.

Look at R squared


5. Develop a Baseline Model

This model should only use real MVP features, at the beginning.

Target variable - Market Value
Features
* Age
* Year
* Position
* Club 

Time horizon.

Predict for the June (month 6) and December (month 12)

Scope - only for players currently in the Premier League

Create - Horizon data

Test / train split - Choose base on year


Because this is based on players age 

6. Iterate and Improve

Arima isn't suggeasted

7. Deploy Monitor and Maintain 

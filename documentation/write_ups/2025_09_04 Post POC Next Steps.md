
The POC gets decent results as explained in INSERT LINK (2025_08_22 POC VERSION etc)

There a lot of things that can be done to improve this and that would need to be improved to put it into production:

The priority should be the first two points within Model Improvements


### Model Improvements

* Currently we only take the last years data, for example number of goals / value and predict the next year within the model.
This should be changed

* Currently there is a cap where if you're above 32 then your value must be 80% or less than the last year, this should be tried for alternatives, where maybe there should be a one for 90% or less when 30, 31 for example.
This could be different by position as well, for example goalkeepers peak at older ages
We shouldn't aim for the model to be improved where this isn't needed for many iterations

We could alternatively try taking age out of the model then doing manual adjustments afterwards.

* Other algorithms / ensemble - e.g Random Forest

* Remove inflation then add at the end 

* Add features - e.g injury record

* Try running on all players rather than just premier league


### Analysis

* General improve / tidy of output framework
* Saving of Shapley values so it's able to create plots of that 

### Refactor

* Save the last few versions of the outputs / set save guards
* Set versions within Github 
* Update version log 
* Remove non required run history
* Speed up running of streamlit
* Save models as pickles / don't need to run the model each time
* Create more shared code within functions
* Set more within a config, which should be tracked using something like MLflow

### Deployment

* Try on further changeover years, where there would need to be data checks / monitoring on data quality
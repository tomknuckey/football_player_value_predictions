## 14th June

### Where I'm at

I've got a very basic MVP that takes predicts next years value


### Next Steps

* Create Git Repo
* Create Venv
* Planning
* Pivot into Time Series 
* Normalisation  / Standardisation
* Creation of more features
    Year - inflation
* Create evaluation format 
* Link to transfers
* Try other models

## Phase 0 - Finished

* Some Planning
* Some EDA
* Basic Model Created

##  Phase 1

Change to Time Series - Predict Transfer Value for the next 3 years, still using the same features of current value and age. - DONE
This will probably be quite a baseline model.

### Steps.

* Determine which model
* Implement it

### Other Work

* Plan and implement Normalisation / Standaridisation 

This isn't needed for decision trees

* Create Venv  - DONE


## Phase 2

Create and test high priority features.

### Steps.

* Year
* Position 
* Sub Position
* Years Left on Contract

### Other Work

* Create Model Evaluation Method (Spreadsheet?)
* Ensure all models are in .py files
* Git Repo creation 



# Phase 3 

Try in Linear regression.

Ensure not overfiting

Change age to linear https://chatgpt.com/c/6878cead-c140-8005-a77d-a75da41440e1

df["age_from_peak"] = (df["age"] - 25) ** 2

df["age_group"] = pd.cut(
    df["age"],
    bins=[15, 21, 27, 34, 45],
    labels=["Young", "Peak", "Post-Peak", "Veteran"]
)



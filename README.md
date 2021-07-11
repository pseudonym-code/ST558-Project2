"# ST558_Project-2" 

This repository is used to create bike sharing summary reports for bike rentals between 2011 and 2012 in the Capital bikeshare system in Washington, DC. There is a separate report for each day.
   
* The analysis for [Monday is available here](Monday.md).
* The analysis for [Tuesday is available here](Tuesday.md).
* The analysis for [Wednesday is available here](Wednesday.md).
* The analysis for [Thursday is available here](Thursday.md).
* The analysis for [Friday is available here](Friday.md).
* The analysis for [Saturday is available here](Saturday.md).
* The analysis for [Sunday is available here](Sunday.md).

The packages used for this analysis are below:

* rmarkdown
* dplyr
* tidyverse
* knitr
* readr
* parallel
* MuMIn
* modelr
* caret
* formatR  
* randomForest
   
Finally, the code used to automate the creation of the daily reports is:
```{r, message=F}
library(knitr)
library(rmarkdown)
library(tidyverse)
data <- read_csv(file="./Bike-Sharing-Dataset/day.csv",col_names = TRUE)
daysFile <- paste0(unique(weekdays(data$dteday[2:8])),".md")
dayNum <- unique(data$weekday[2:8])
params = lapply(dayNum, FUN = function(x){list(day = x)})
reports <- tibble(daysFile, params)
apply(reports, MARGIN = 1,
        FUN = function(x){
          render(input = "Project 2.Rmd", output_file = x[[1]], params = x[[2]])
        })
render('Project 2.Rmd', output_format = "github_document", output_file = "README.md")
```

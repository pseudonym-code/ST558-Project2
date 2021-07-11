ST 558 Project 2
================
Yu Bai & Lee Pixton
7/11/2021

# Introduction

People are constantly finding new and innovative ways to help people get
around town, especially in large cities. One of these more recent
methods is through bike sharing. Through technology and smart phones,
the whole process of renting a bike, from membership, rental and return
has become automatic. These systems allow users to easily rent a bike
from a location nearby and return it back to another location, without
having to return to the pick up spot.  
The data we will be exploring contains the hourly and daily counts of
bike rentals between 2011 and 2012 in the Capital bikeshare system in
Washington, DC. We will be focused on predicting the number of casual
users, and be exploring the following variables:

-   `casual` - count of casual users
-   `season` - winter, spring, summer or fall
-   `workingday` - whether the day is a work day (not a holiday or
    weekend)
-   `weathersit` - weather situation (clear, misty, light precipitation,
    heavy precipitation)
-   `atemp` - Temperature feel in degrees Celsius
-   `hum` - humidity level

We will use both linear regression and ensemble learning methods to
predict the casual user count with the above variables.

## Packages

The packages that will be used in this analysis are below.

``` r
library(rmarkdown)
library(dplyr)
library(tidyverse)
library(knitr)
library(readr)
library(parallel)
library(MuMIn)
library(modelr)
library(caret)
library(formatR)
library(randomForest)
```

# Data

## Read csv data

First we read in the data from a csv file.

``` r
data <- read_csv(file = "./Bike-Sharing-Dataset/day.csv",
    col_names = TRUE)

# Select Monday
p2 <- data %>%
    filter(weekday == params$day)
```

## Split data into train data (70%) and test data (30%)

Next, we set the categorical variables to factors and create a test and
train set.

``` r
# Set categorical variables to factors
p2$season <- factor(p2$season, levels = c(1, 2, 3,
    4), labels = c("Spring", "Summer", "Fall", "Winter"))
p2$yr <- factor(p2$yr, levels = c(0, 1), labels = c("2011",
    "2012"))
p2$holiday <- factor(p2$holiday, levels = c(0, 1),
    labels = c("Not holiday", "Holiday"))
p2$weekday <- factor(p2$weekday, levels = c(0, 1, 2,
    3, 4, 5, 6), labels = c("Sunday", "Monday", "Tuesday",
    "Wednesday", "Thursday", "Friday", "Saturday"))
p2$workingday <- factor(p2$workingday, levels = c(0,
    1), labels = c("Not workingday", "Workingday"))
p2$weathersit <- factor(p2$weathersit, levels = c(1,
    2, 3, 4), labels = c("Clear", "Mist cloudy", "Light snow/rain",
    "Heavy rain/snow"))
p2$mnth <- factor(p2$mnth)

# Create train and test sets
train <- sample(1:nrow(p2), size = nrow(p2) * 0.7)
test <- dplyr::setdiff(1:nrow(p2), train)
p2Train <- p2[train, ]
p2Test <- p2[test, ]
```

## Summarizations

After reading in the data, we can now examine our training dataset. As
mentioned above, we will be specifically looking at the casual users of
bike sharing in this analysis.

### Casual Users by Season

The table below shows the distribution of casual users across four
seasons. It helps us see a season effect.

``` r
### count of casual users by season
s1 <- p2Train %>%
    group_by(season) %>%
    summarize(avg_casual = mean(casual, na.rm = TRUE),
        sd_casual = sd(casual, na.rm = TRUE), min_casual = min(casual,
            na.rm = TRUE), max_casual = max(casual,
            na.rm = TRUE))

kable(s1)
```

| season | avg\_casual | sd\_casual | min\_casual | max\_casual |
|:-------|------------:|-----------:|------------:|------------:|
| Spring |    260.5333 |   210.4973 |          42 |         838 |
| Summer |    864.4211 |   535.6975 |         195 |        2557 |
| Fall   |   1126.1000 |   542.6129 |         568 |        3065 |
| Winter |    581.7368 |   380.6601 |           2 |        1514 |

The box-plot displays the minimum, maximum, first quartile, third
quartile, median and outliers for casual users across seasons. It helps
us see difference in casual users over seasons.

``` r
# Boxplot for count of casual users by season
g1 <- ggplot(data = p2Train, aes(x = casual, y = season))
g1 + geom_boxplot() + labs(subtitle = "Boxplot for count of casual users by season",
    x = "Count of casual users", y = "Season") + guides(fill = guide_legend(title = NULL))
```

![](Monday_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

The table below shows the distribution of casual users in workingday and
non-workingday across four seasons. It helps us see an interaction of
season and workingday effect.

``` r
### count of casual users by workingday and
### season
s4 <- p2Train %>%
    group_by(season, workingday) %>%
    summarize(avg_casual = mean(casual, na.rm = TRUE),
        sd_casual = sd(casual, na.rm = TRUE), min_casual = min(casual,
            na.rm = TRUE), max_casual = max(casual,
            na.rm = TRUE))
kable(s4)
```

| season | workingday     | avg\_casual | sd\_casual | min\_casual | max\_casual |
|:-------|:---------------|------------:|-----------:|------------:|------------:|
| Spring | Not workingday |    317.6000 |   138.9291 |         195 |         502 |
| Spring | Workingday     |    232.0000 |   240.0690 |          42 |         838 |
| Summer | Not workingday |   1768.0000 |   705.4722 |        1198 |        2557 |
| Summer | Workingday     |    695.0000 |   289.9989 |         195 |        1208 |
| Fall   | Not workingday |   2088.6667 |   920.7499 |        1236 |        3065 |
| Fall   | Workingday     |    956.2353 |   198.1892 |         568 |        1233 |
| Winter | Not workingday |   1161.6667 |   324.8636 |         874 |        1514 |
| Winter | Workingday     |    473.0000 |   282.7284 |           2 |        1001 |

The graph below shows the distribution of casual users in workingday and
non-workingday across four seasons. It helps us compare the casual users
by workingday over four seasons.

``` r
# Barplot for count of casual users by workingday
# and season
g <- ggplot(data = p2Train, aes(x = casual, y = season))
g + geom_boxplot() + labs(subtitle = "Boxplot for count of casual users by workingday and season",
    x = "Count of casual users", y = "Season") + guides(fill = guide_legend(title = NULL)) +
    facet_wrap(~workingday)
```

![](Monday_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

Here is another way to compare the count of casual users in workingday
and non-workingday across four seasons. It helps us understand the
difference in the casual users by workingday over four seasons.

``` r
# Scatterplot for casual by workingday and season
g1 <- ggplot(data = p2Train, aes(x = casual, y = workingday,
    group = season))
g1 + geom_point(aes(color = season)) + labs(title = "Count of Casual Users by Working Day and Season",
    x = "Count of casual users", y = "Working Day")
```

![](Monday_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

### Casual Users by Temperature

The graph below shows the relationship between the count of casual users
and temperature. It helps us understand whether there is a linear
association between the two variables.

``` r
# Graph: temperature feel and count of casual
# users

g4 <- ggplot(data = p2Train, aes(x = atemp, y = casual))
g4 + geom_point(aes(x = temp, y = casual), size = 3) +
    geom_smooth(data = p2Train, formula = y ~ x, method = lm,
        col = "Green") + labs(subtitle = "Temperature Feel and count of casual users",
    x = "Temperature Feel", y = "Count of casual users") +
    guides(fill = guide_legend(title = NULL))
```

![](Monday_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

### Casual Users by Humidity

The graph below shows the relationship between the count of casual users
and humidity. It helps us understand whether there is a linear
association between the two variables.

``` r
# Graph: humidity and count of casual users

g5 <- ggplot(data = p2Train, aes(x = hum, y = casual))
g5 + geom_point(aes(x = hum, y = casual), size = 3) +
    geom_smooth(data = p2Train, formula = y ~ x, method = lm,
        col = "Green") + labs(subtitle = "Humidity and count of casual users",
    x = "Normalized humidity", y = "Count of casual users") +
    guides(fill = guide_legend(title = NULL))
```

![](Monday_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

### Weather Situations by Season

The table below shows the weather status over seasons.

``` r
### contingency table
table(p2Train$season, p2Train$weathersit)
```

    ##         
    ##          Clear Mist cloudy Light snow/rain Heavy rain/snow
    ##   Spring    10           5               0               0
    ##   Summer    11           8               0               0
    ##   Fall      12           8               0               0
    ##   Winter    11           6               2               0

### Casual Users by Weather Situation

The table below shows the distribution of the count of casual users over
three weather situations. It helps us understand the weather effect.

``` r
### count of casual users by Weather Situation
s6 <- p2Train %>%
    group_by(weathersit) %>%
    summarize(avg = mean(casual, na.rm = TRUE), sd = sd(casual,
        na.rm = TRUE), min = min(casual, na.rm = TRUE),
        max = max(casual, na.rm = TRUE))
kable(s6)
```

| weathersit      |      avg |       sd | min |  max |
|:----------------|---------:|---------:|----:|-----:|
| Clear           | 777.8409 | 515.2536 |  86 | 2557 |
| Mist cloudy     | 720.7407 | 584.5461 |  42 | 3065 |
| Light snow/rain | 111.0000 | 154.1493 |   2 |  220 |

### Casual Users by Weather Situations and Seasons

The graph below shows the count of casual users over seasons by three
weather situations. It helps us understand whether there is an
interaction between weather and season.

``` r
# Count of casual users by weather and season
g2 <- ggplot(p2Train, aes(x = season))
g2 + geom_bar(aes(fill = casual, position = "dodge",
    color = season), fill = "white") + labs(title = "Count of Casual Users by Season and Weather",
    y = "Count of Casual Users", x = "Season") + theme(legend.title = element_blank(),
    axis.text.x = element_blank()) + facet_wrap(p2Train$weathersit)
```

![](Monday_files/figure-gfm/unnamed-chunk-13-1.png)<!-- -->

### Casual Users by Humidity, Temperature, Weather Situations and Seasons

The graph below shows the effects of humidity and temperature on the
count of casual users over seasons. It helps us understand how multiple
variables affect casual users.

``` r
# Graph: humidity and count of casual users

g3 <- ggplot(p2Train, aes(x = atemp, y = hum))
g3 + geom_point(aes(size = casual, color = season)) +
    labs(title = "Casual Users by Temperature Feel and Humidty",
        x = "Temperature Feel", y = "Humidty") + guides(fill = guide_legend(title = NULL))
```

![](Monday_files/figure-gfm/unnamed-chunk-14-1.png)<!-- -->

## Modeling

Linear regression is used to estimate the association between a response
variable and one or more predictors by using a linear approach. Four
models are built for the analysis. The first one is a linear model,
which includes 5 predictors (`season`, `workingday`, `weathersit`,
`atemp`, `hum`). The second one is a random forest model. The third one
is once again a linear model, which includes only 4 predictors
(`season`, `workingday`, `weathersit`, `atemp`), removing the `hum`
variable. The fourth model is a boosted tree model.

``` r
# Model 1: Linear model with all 5 predictor
# variables
f_m1 <- as.formula("casual~season+workingday+weathersit+atemp+hum")

fit_m1 <- train(f_m1, data = p2Train, method = "lm",
    preProcess = c("center", "scale"), trControl = trainControl(method = "cv",
        number = 10))

summary(fit_m1)
```

    ## 
    ## Call:
    ## lm(formula = .outcome ~ ., data = dat)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -521.15 -159.08  -48.72  138.84 1332.00 
    ## 
    ## Coefficients: (1 not defined because of singularities)
    ##                             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                   738.45      37.01  19.955  < 2e-16 ***
    ## seasonSummer                  106.39      69.86   1.523    0.133    
    ## seasonFall                    127.65      88.46   1.443    0.154    
    ## seasonWinter                   98.89      60.27   1.641    0.106    
    ## workingdayWorkingday         -250.53      38.60  -6.489 1.45e-08 ***
    ## `weathersitMist cloudy`        28.30      46.86   0.604    0.548    
    ## `weathersitLight snow/rain`   -46.99      42.48  -1.106    0.273    
    ## `weathersitHeavy rain/snow`       NA         NA      NA       NA    
    ## atemp                         315.00      71.41   4.411 4.02e-05 ***
    ## hum                           -73.86      52.33  -1.412    0.163    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 316.2 on 64 degrees of freedom
    ## Multiple R-squared:  0.698,  Adjusted R-squared:  0.6603 
    ## F-statistic: 18.49 on 8 and 64 DF,  p-value: 5.294e-14

Random forests are a learning approach for classification and regression
problems to build a predictive model. This model is made of many
decision trees.

``` r
# Define trainControl
trctrl <- trainControl(method = "repeatedcv", number = 10,
    repeats = 3)

# Model 2: Ensemble tree model (Random forest)
fit_m2 <- train(f_m1, data = p2Train, method = "rf",
    trControl = trctrl, preProcess = c("center", "scale"),
    tuneGrid = data.frame(mtry = 1))

fit_m2
```

    ## Random Forest 
    ## 
    ## 73 samples
    ##  5 predictor
    ## 
    ## Pre-processing: centered (9), scaled (9) 
    ## Resampling: Cross-Validated (10 fold, repeated 3 times) 
    ## Summary of sample sizes: 66, 66, 65, 65, 65, 65, ... 
    ## Resampling results:
    ## 
    ##   RMSE     Rsquared  MAE     
    ##   395.516  0.662069  301.8651
    ## 
    ## Tuning parameter 'mtry' was held constant at a value of 1

Now we can fit the next linear model, removing the `hum` variable as it
was not significant in the earlier model.

``` r
# Model 3: Linear model after dropping humidity
# from the first model
fit_m3 <- train(casual ~ season + workingday + weathersit +
    atemp, data = p2Train, method = "lm", preProcess = c("center",
    "scale"), trControl = trainControl(method = "cv",
    number = 10))

summary(fit_m3)
```

    ## 
    ## Call:
    ## lm(formula = .outcome ~ ., data = dat)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -602.99 -165.54  -50.12  121.50 1392.99 
    ## 
    ## Coefficients: (1 not defined because of singularities)
    ##                             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                   738.45      37.29  19.804  < 2e-16 ***
    ## seasonSummer                  113.80      70.19   1.621   0.1098    
    ## seasonFall                    141.33      88.59   1.595   0.1155    
    ## seasonWinter                   81.99      59.52   1.378   0.1731    
    ## workingdayWorkingday         -247.11      38.82  -6.365 2.26e-08 ***
    ## `weathersitMist cloudy`       -10.53      38.22  -0.276   0.7838    
    ## `weathersitLight snow/rain`   -69.98      39.54  -1.770   0.0814 .  
    ## `weathersitHeavy rain/snow`       NA         NA      NA       NA    
    ## atemp                         285.66      68.84   4.150 9.87e-05 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 318.6 on 65 degrees of freedom
    ## Multiple R-squared:  0.6886, Adjusted R-squared:  0.6551 
    ## F-statistic: 20.54 on 7 and 65 DF,  p-value: 2.859e-14

Boosting tree models is a general approach that consists of growing the
trees sequentially. This means that each tree is based off of the
previous one, and predictions are updated along the way. In essence, we
are using several weaker tree models sequentially in order to create a
strong tree model. The errors of previous models are minimized and the
next model is then “boosted” to ideally improve the accuracy of
predictions and reduce error. Here we will use all five of our variables
of interest.

``` r
# Model 4: Ensemble tree model (boosted tree)
gbmGrid <- expand.grid(interaction.depth = 4, n.trees = 1000,
    shrinkage = 0.1, n.minobsinnode = 2)

fit_m4 <- train(casual ~ season + workingday + weathersit +
    atemp + hum, data = p2Train, method = "gbm", trControl = trctrl,
    preProcess = c("center", "scale"), verbose = FALSE,
    tuneGrid = gbmGrid)

fit_m4
```

    ## Stochastic Gradient Boosting 
    ## 
    ## 73 samples
    ##  5 predictor
    ## 
    ## Pre-processing: centered (9), scaled (9) 
    ## Resampling: Cross-Validated (10 fold, repeated 3 times) 
    ## Summary of sample sizes: 66, 66, 66, 66, 65, 66, ... 
    ## Resampling results:
    ## 
    ##   RMSE      Rsquared   MAE     
    ##   278.6472  0.7978239  204.2917
    ## 
    ## Tuning parameter 'n.trees' was held constant at a value of 1000
    ## Tuning
    ##  parameter 'shrinkage' was held constant at a value of 0.1
    ## Tuning
    ##  parameter 'n.minobsinnode' was held constant at a value of 2

## Model Comparison

After fitting the models, we can now compare them using multiple
methods. We will display each model’s RMSE, R-squared, and MAE below.

``` r
result <- as.matrix(t(data.frame(RMSE = c(fit_m1$results[[2]],
    fit_m2$results[[2]], fit_m3$results[[2]], fit_m4$results[[5]]),
    Rsquared = c(fit_m1$results[[3]], fit_m2$results[[3]],
        fit_m3$results[[3]], fit_m4$results[[6]]),
    MAE = c(fit_m1$results[[4]], fit_m2$results[[4]],
        fit_m3$results[[4]], fit_m4$results[[7]]))))

colnames(result) <- c("Model1", "Model2", "Model3",
    "Model4")

kable(result)
```

|          |      Model1 |     Model2 |      Model3 |      Model4 |
|:---------|------------:|-----------:|------------:|------------:|
| RMSE     | 316.8653757 | 395.516004 | 305.5981483 | 278.6471616 |
| Rsquared |   0.6990728 |   0.662069 |   0.6927899 |   0.7978239 |
| MAE      | 236.4359667 | 301.865135 | 229.0790008 | 204.2917337 |

## Model selection

Now we can select a best model fit for this data.

``` r
RMSE_min <- min(fit_m1$results[[2]], fit_m2$results[[2]],
    fit_m3$results[[2]], fit_m4$results[[5]])

if (fit_m1$results[[2]] == RMSE_min) {
    model <- 1
} else if (fit_m2$results[[2]] == RMSE_min) {
    model <- 2
} else if (fit_m3$results[[2]] == RMSE_min) {
    model <- 3
} else if (fit_m4$results[[5]] == RMSE_min) {
    model <- 4
}
```

The best model we will use is the one with smallest RMSE. Based on the
table above, the best model is Model 4

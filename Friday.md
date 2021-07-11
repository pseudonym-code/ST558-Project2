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

# Data

## Read csv data

First we read in the data from a csv file.

``` r
getwd()
```

    ## [1] "D:/Documents/NCSU/ST558/ST558_Project-2"

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

``` r
### count of casual users by season
s1 <- p2Train %>%
    group_by(season) %>%
    summarize(avg_casual = mean(casual, na.rm = TRUE),
        sd_casual = sd(casual, na.rm = TRUE), min_casual = min(casual,
            na.rm = TRUE), max_casual = max(casual,
            na.rm = TRUE))

kable(s1, caption = "Count of casual users by season")
```

| season | avg\_casual | sd\_casual | min\_casual | max\_casual |
|:-------|------------:|-----------:|------------:|------------:|
| Spring |    280.5714 |   191.0524 |          38 |         644 |
| Summer |    928.7895 |   435.0337 |         172 |        1563 |
| Fall   |   1035.9524 |   291.7448 |         417 |        1511 |
| Winter |    765.0000 |   409.0553 |         245 |        1603 |

Count of casual users by season

``` r
## Graph Boxplot for count of casual users by
## season
g1 <- ggplot(data = p2Train, aes(x = casual, y = season))
g1 + geom_boxplot() + labs(subtitle = "Boxplot for count of casual users by season",
    x = "Count of casual users", y = "Season") + guides(fill = guide_legend(title = NULL))
```

![](Friday_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

### Casual Users by Year

``` r
### count of casual users by year
s2 <- p2Train %>%
    group_by(yr) %>%
    summarize(avg_casual = mean(casual, na.rm = TRUE),
        sd_casual = sd(casual, na.rm = TRUE), min_casual = min(casual,
            na.rm = TRUE), max_casual = max(casual,
            na.rm = TRUE))
kable(s2, caption = "Count of casual users by year")
```

| yr   | avg\_casual | sd\_casual | min\_casual | max\_casual |
|:-----|------------:|-----------:|------------:|------------:|
| 2011 |    598.8387 |   346.6632 |          38 |        1246 |
| 2012 |    939.9024 |   448.5917 |         115 |        1603 |

Count of casual users by year

``` r
## Graph Boxplot for count of casual users by
## year
g2 <- ggplot(data = p2Train, aes(x = casual, y = yr))
g2 + geom_boxplot() + labs(subtitle = "Boxplot for count of casual users by year",
    x = "Count of casual users", y = "Year") + guides(fill = guide_legend(title = NULL))
```

![](Friday_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

### Casual Users by Both Year and Season

``` r
### count of casual users by year and season
s3 <- p2Train %>%
    group_by(yr, season) %>%
    summarize(avg_casual = mean(casual, na.rm = TRUE),
        sd_casual = sd(casual, na.rm = TRUE), min_casual = min(casual,
            na.rm = TRUE), max_casual = max(casual,
            na.rm = TRUE))
kable(s3, caption = "Count of casual users by year and season")
```

| yr   | season | avg\_casual | sd\_casual | min\_casual | max\_casual |
|:-----|:-------|------------:|-----------:|------------:|------------:|
| 2011 | Spring |    198.0000 |   195.7560 |          38 |         579 |
| 2011 | Summer |    628.2222 |   319.1292 |         172 |         898 |
| 2011 | Fall   |    848.1111 |   248.9129 |         417 |        1246 |
| 2011 | Winter |    584.1429 |   308.4118 |         245 |        1095 |
| 2012 | Spring |    342.5000 |   173.7494 |         115 |         644 |
| 2012 | Summer |   1199.3000 |   341.0774 |         533 |        1563 |
| 2012 | Fall   |   1176.8333 |   243.2566 |         747 |        1511 |
| 2012 | Winter |    880.0909 |   435.7946 |         349 |        1603 |

Count of casual users by year and season

``` r
# Barplot for count of casual users by year and
# season
g3 <- ggplot(data = p2Train, aes(x = casual, y = season))
g3 + geom_boxplot() + labs(subtitle = "Boxplot for count of casual users by year and year",
    x = "Count of casual users", y = "Season") + guides(fill = guide_legend(title = NULL)) +
    facet_wrap(~yr)
```

![](Friday_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

### Casual Users by Temperature Feel

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

![](Friday_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

### Casual Users by Humidity

``` r
# Graph: humidity and count of casual users

g5 <- ggplot(data = p2Train, aes(x = hum, y = casual))
g5 + geom_point(aes(x = hum, y = casual), size = 3) +
    geom_smooth(data = p2Train, formula = y ~ x, method = lm,
        col = "Green") + labs(subtitle = "Humidity and count of casual users",
    x = "Normalized humidity", y = "Count of casual users") +
    guides(fill = guide_legend(title = NULL))
```

![](Friday_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->

### Casual Users by Working Day and Season

``` r
### count of casual users by working day and
### season
s4 <- p2Train %>%
    group_by(workingday) %>%
    summarize(avg = mean(casual, na.rm = TRUE), sd = sd(casual,
        na.rm = TRUE), min = min(casual, na.rm = TRUE),
        max = max(casual, na.rm = TRUE))

kable(s4, caption = "Casual Users by Working Day")
```

| workingday     |      avg |       sd | min |  max |
|:---------------|---------:|---------:|----:|-----:|
| Not workingday | 642.0000 |       NA | 642 |  642 |
| Workingday     | 795.1831 | 442.1531 |  38 | 1603 |

Casual Users by Working Day

``` r
# Scatterplot for casual by workingday and season
g6 <- ggplot(data = p2Train, aes(x = casual, y = workingday,
    group = season))
g6 + geom_point(aes(color = season)) + labs(title = "Count of Casual Users by Working Day and Season",
    x = "Count of casual users", y = "Working Day")
```

![](Friday_files/figure-gfm/unnamed-chunk-13-1.png)<!-- -->

### Casual Users by Weather

``` r
### count of casual users by weather situation
s5 <- p2Train %>%
    group_by(weathersit) %>%
    summarize(avg = mean(casual, na.rm = TRUE), sd = sd(casual,
        na.rm = TRUE), min = min(casual, na.rm = TRUE),
        max = max(casual, na.rm = TRUE))
kable(s5, caption = "Casual Users by Weather Situation")
```

| weathersit  |      avg |       sd | min |  max |
|:------------|---------:|---------:|----:|-----:|
| Clear       | 940.1628 | 393.0620 | 149 | 1603 |
| Mist cloudy | 574.9310 | 418.7086 |  38 | 1511 |

Casual Users by Weather Situation

### Full Count by Season

``` r
### Complete count of casual users by season
s6 <- p2Train %>%
    group_by(season) %>%
    summarize(count = sum(casual, na.rm = TRUE))
kable(s6, caption = "Count of Casual Users by Season")
```

| season | count |
|:-------|------:|
| Spring |  3928 |
| Summer | 17647 |
| Fall   | 21755 |
| Winter | 13770 |

Count of Casual Users by Season

### Casual Users by Weather and Season

``` r
# Count of casual users by weather and season
g7 <- ggplot(p2Train, aes(x = season))
g7 + geom_bar(aes(fill = casual, position = "dodge",
    color = season), fill = "white") + labs(title = "Count of Casual Users by Season and Weather",
    y = "Count of Casual Users", x = "Season") + theme(legend.title = element_blank(),
    axis.text.x = element_blank()) + facet_wrap(p2Train$weathersit)
```

![](Friday_files/figure-gfm/unnamed-chunk-16-1.png)<!-- -->

### Casual Users by Humidity, Temperature, and Season

``` r
# Graph: humidity and count of casual users

g8 <- ggplot(p2Train, aes(x = atemp, y = hum))
g8 + geom_point(aes(size = casual, color = season)) +
    labs(title = "Casual Users by Temperature Feel and Humidty",
        x = "Temperature Feel", y = "Humidty") + guides(fill = guide_legend(title = NULL))
```

![](Friday_files/figure-gfm/unnamed-chunk-17-1.png)<!-- -->

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
    ## -639.56 -164.06  -38.72  165.35  835.66 
    ## 
    ## Coefficients: (2 not defined because of singularities)
    ##                             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                   793.06      34.78  22.800  < 2e-16 ***
    ## seasonSummer                  132.78      57.17   2.323   0.0234 *  
    ## seasonFall                     64.66      78.02   0.829   0.4103    
    ## seasonWinter                  127.59      50.14   2.545   0.0134 *  
    ## workingdayWorkingday           19.46      36.39   0.535   0.5946    
    ## `weathersitMist cloudy`       -43.37      46.10  -0.941   0.3504    
    ## `weathersitLight snow/rain`       NA         NA      NA       NA    
    ## `weathersitHeavy rain/snow`       NA         NA      NA       NA    
    ## atemp                         266.51      63.44   4.201 8.39e-05 ***
    ## hum                           -90.94      43.85  -2.074   0.0421 *  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 295.1 on 64 degrees of freedom
    ## Multiple R-squared:  0.5933, Adjusted R-squared:  0.5488 
    ## F-statistic: 13.34 on 7 and 64 DF,  p-value: 1.783e-10

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
    ## 72 samples
    ##  5 predictor
    ## 
    ## Pre-processing: centered (9), scaled (9) 
    ## Resampling: Cross-Validated (10 fold, repeated 3 times) 
    ## Summary of sample sizes: 64, 64, 64, 66, 64, 66, ... 
    ## Resampling results:
    ## 
    ##   RMSE      Rsquared   MAE     
    ##   370.0575  0.5970599  311.5854
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
    ## -606.39 -164.79  -49.88  178.55  834.73 
    ## 
    ## Coefficients: (2 not defined because of singularities)
    ##                             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                   793.06      35.66  22.242  < 2e-16 ***
    ## seasonSummer                  134.12      58.60   2.289 0.025349 *  
    ## seasonFall                     77.76      79.71   0.976 0.332909    
    ## seasonWinter                  114.90      51.01   2.252 0.027682 *  
    ## workingdayWorkingday           30.44      36.90   0.825 0.412512    
    ## `weathersitMist cloudy`       -99.81      38.14  -2.617 0.011020 *  
    ## `weathersitLight snow/rain`       NA         NA      NA       NA    
    ## `weathersitHeavy rain/snow`       NA         NA      NA       NA    
    ## atemp                         237.48      63.43   3.744 0.000386 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 302.5 on 65 degrees of freedom
    ## Multiple R-squared:  0.566,  Adjusted R-squared:  0.5259 
    ## F-statistic: 14.13 on 6 and 65 DF,  p-value: 3.209e-10

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
    ## 72 samples
    ##  5 predictor
    ## 
    ## Pre-processing: centered (9), scaled (9) 
    ## Resampling: Cross-Validated (10 fold, repeated 3 times) 
    ## Summary of sample sizes: 65, 64, 65, 64, 66, 67, ... 
    ## Resampling results:
    ## 
    ##   RMSE      Rsquared   MAE     
    ##   363.0274  0.4276206  299.8404
    ## 
    ## Tuning parameter 'n.trees' was held constant at a value of 1000
    ## Tuning parameter 'interaction.depth' was held constant at a value of 4
    ## Tuning parameter 'shrinkage' was held constant at a
    ##  value of 0.1
    ## Tuning parameter 'n.minobsinnode' was held constant at a value of 2

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

|          |      Model1 |      Model2 |      Model3 |      Model4 |
|:---------|------------:|------------:|------------:|------------:|
| RMSE     | 291.5380620 | 370.0575379 | 303.9716753 | 363.0274349 |
| Rsquared |   0.5797647 |   0.5970599 |   0.5991659 |   0.4276206 |
| MAE      | 239.1533178 | 311.5853550 | 251.6334810 | 299.8404342 |

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
table above, the best model is Model 1

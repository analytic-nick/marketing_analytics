###Load Packages
library(readr)
pacman::p_load(minpack.lm)

### Load Data
data<-read_csv("https://raw.githubusercontent.com/analytic-nick/marketing_analytics/main/mmm/data/de_simulated_data.csv")


##############################Saturation Effect
# Load libraries
library(ggplot2)
library(plotly)

# Hill function definition
saturation_hill <- function(x, alpha, gamma, Vmax = 1000, x_marginal = NULL) {
  inflexion <- c(range(x) %*% c(1 - gamma, gamma)) # linear interpolation with scalar product
  if (is.null(x_marginal)) {
    x_scurve <- Vmax * (x^alpha / (x^alpha + inflexion^alpha))
  } else {
    x_scurve <- Vmax * (x_marginal^alpha / (x_marginal^alpha + inflexion^alpha))
  }
  return(x_scurve)
}

# Graph parameters
x <- seq(0, 200000, length.out = 100)
alphas <- c(1, 2, 3)
gammas <- c(0.1, 0.5, 0.9)

# Simulating data
data <- data.frame()
for (alpha in alphas) {
  for (gamma in gammas) {
    response <- saturation_hill(x, alpha, gamma)
    temp_data <- data.frame(Spending = x, Response = response, Alpha = alpha, Gamma = gamma)
    data <- rbind(data, temp_data)
  }
}


# Graph
data$Colors<-paste(data$Gamma, data$Alpha,sep =",")

p <- ggplot(data, aes(x = Spending, y = Response, color = Colors)) +
  geom_line(size = 0.8) +
  scale_color_manual(values=c("#005f73", "#0a9396", "#94d2bd","#e9d8a6", "#ee9b00","#ca6702", "#bb3e03", "#ae2012", "#9b2226")) + # Utilizzo di una palette di colori accattivante
  labs(
    title = "Saturation Curve - Hill Function",
    subtitle = "Marketing Spending vs Response",
    x = "Marketing Spending",
    y = "Response",
    color = "Gamma & Alpha",
    linetype = "Gamma"
  ) +
  scale_y_continuous(limits = c(0, 1000)) + # Formatting axes
  theme_minimal(base_size = 10) + # Minimal theme with increased font size
  theme(
    plot.title = element_text(hjust = 0.5, size = 15, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, size = 10),
    axis.title = element_text(size = 10, face = "bold"),
    legend.title = element_text(size = 10, face = "bold"),
    legend.text = element_text(size = 10),
    legend.position = "top" # Positioning legend on the top
  )

##plot the graph
ggplotly(p)


#############################################Calculate Adstock



### Create Adstock Function

# adstock function
adstock<-function(x,rate=0){
  return(as.numeric(stats::filter(x=x,filter=rate,method="recursive")))
}

### Create Adstock Optimization Function

AdstockRateMV <- function(Impact, Ads, maxiter = 100){
  # parameter names
  params = letters[2:(ncol(Ads)+1)]
  # ad variable names
  ads = paste0("ad_", params)
  # rate variable names
  rates = paste0("rate_", params)
  # create partial formula
  param_fm = paste(
    paste(params, "*adstock(", ads,  ",", rates, ")", sep = ""),
    collapse = " + "
  )
  # create whole formula
  fm = as.formula(paste("Impact ~ a +", param_fm))
  # starting values for nls
  start = c(rep(1, length(params) + 1), rep(.1, length(rates)))
  names(start) = c("a", params, rates)
  # input data
  Ads_df = Ads
  names(Ads_df) = ads
  Data = cbind(Impact, Ads_df)
  # fit model
  modFit <- nls(data = Data, fm, start = start, control = nls.control(maxiter = maxiter, warnOnly = T))
  # if all decay rates greater than 0, done. If not, use a constrained nls model (with lower and upper parameter limits)
  if(!all(summary(modFit)$coefficients[rates, 1] > 0)){
    library(minpack.lm)
    lower = c(rep(-Inf, length(params) + 1), rep(0, length(rates)))
    upper = c(rep(Inf, length(params) + 1), rep(1, length(rates)))
    modFit <- nlsLM(fm, data = Data, start = start,
                    lower = lower, upper = upper,
                    control = nls.lm.control(maxiter = maxiter)) 
  }
  # model coefficients
  AdstockInt = round(summary(modFit)$coefficients[1, 1])
  AdstockCoef = round(summary(modFit)$coefficients[params, 1], 2)
  AdstockRate = round(summary(modFit)$coefficients[rates, 1], 2)
  # print formula with coefficients
  param_fm_coefs = paste(
    paste(round(AdstockCoef, 2), " * adstock(", names(Ads),  ", ", round(AdstockRate, 2), ")", sep = ""),
    collapse = " + "
  )
  fm_coefs = as.formula(paste("Impact ~ ", AdstockInt, " +", param_fm_coefs))
  # rename rates with original variable names
  names(AdstockRate) = paste0("rate_", names(Ads))
  # calculate percent error
  mape = mean(abs((Impact-predict(modFit))/Impact) * 100)
  # return outputs
  return(list(fm = fm_coefs, base = AdstockInt, rates = AdstockRate, mape = mape))
}


### Create Dataframe of all media vars
media_vars=c("tv_S", "ooh_S", "print_S", "facebook_S", "search_S")
Ads<-data[media_vars]
Impact= as.numeric(unlist(data["revenue"]))

###  Get Optimal Rates
rates<-AdstockRateMV(Impact,Ads)$rates

### Create Empty Adstock Dataframe
List <- colnames(Ads)
adstocks <- matrix(data = NA_integer_, nrow = nrow(Ads), ncol = length(List))
colnames(adstocks)<-colnames(Ads)

### Get Adstock Values

for (i in seq.default(1, length(List))) {
  adstocks[,i] <- adstock(Ads[[i]],rates[i]) 
}

colnames(adstocks) <- paste0(colnames(adstocks),'_adstock')

##############################Add Saturation Effect to adstock
# Load libraries
media_adstock=c("tv_S_adstock", "ooh_S_adstock", "print_S_adstock", "facebook_S_adstock", "search_S_adstock")
alpha=.5
gamma=.5

### Create Empty Adstock Dataframe
List2 <- colnames(adstocks)
sat_adstocks <- matrix(data = NA_integer_, nrow = nrow(Ads), ncol = length(List))
colnames(sat_adstocks)<-colnames(adstocks)

for (i in seq.default(1, length(List2))) {
  sat_adstocks[,i] <- data.table::saturation_hill(adstocks[,i],alpha,gamma) 
}

sat_adstocks<-as.data.frame(sat_adstocks)
as.numeric(unlist(sat_adstocks["tv_S_adstock"]))
p <- ggplot(Ads, aes(x = tv_S, y = as.numeric(unlist(sat_adstocks["tv_S_adstock"])))) + geom_line(size = 0.8)
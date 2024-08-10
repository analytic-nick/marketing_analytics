library(data.table) 
library(stringr) 
library(lubridate) 
library(ggplot2)
library(prophet)
library(readr)
library(glmnet)
library(tidyverse)
library(caret)


###################################################Step 1 : Load Data

data<-read_csv("https://raw.githubusercontent.com/analytic-nick/marketing_analytics/main/mmm/data/de_simulated_data.csv")

#####################Step 2 : Feature Engineering
prophet_data = data
prophet_data$ds=prophet_data$DATE
prophet_data$y=prophet_data$revenue


m<-prophet(prophet_data)
prophet_predict = predict(m,prophet_data)


final_data = data
final_data['trend'] = prophet_predict["trend"]
final_data['season'] = prophet_predict["yearly"]
final_data


######################Calculate Adstock


# adstock function

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
final_data<-cbind(final_data,adstocks)

##########################################################Step 3 : Model Specification

##Specify Model Variables
target = "revenue"
media_channels = c("tv_S_adstock", "ooh_S_adstock", "print_S_adstock", "facebook_S_adstock", "search_S_adstock")
control_features = c( "competitor_sales_B","trend","season")
features = c(control_features,media_channels )

#Set lower and upper limites
lb=c(-Inf, -Inf,-Inf,0, 0,  0,0,0)
ub=c(0,Inf,Inf,Inf, Inf, Inf,Inf,Inf)

#set penalty factor
p.fac <- rep(1,8)
p.fac[c(1,2,4, 5,6 ,7,8)] <- 0
p.fac[c(3)] <- 2



x_train = as.matrix(as.data.frame(final_data[features]))
y_train = final_data[target]
y_train=as.numeric(unlist(y_train))


#####################Step 4 : Cross Validation for Optimal Lambda


cv.out <- cv.glmnet(x_train,y_train,alpha=1,type.measure = "mse" ,lower.limits=lb,upper.limits=ub,penalty.factor = p.fac)



#min value of lambda
lambda_min <- cv.out$lambda.min
#best value of lambda
lambda_1se <- cv.out$lambda.1se
#regression coefficients
coef(cv.out,s=lambda_1se)


#####################Step 5 : Run Model

model.ridge <- glmnet(x_train, 
                      y_train, 
                       is_intercept = True, lambda = lambda_1se,lower.limits=lb,upper.limits=ub)

coefs<-data.frame(coef.name = dimnames(coef(model.ridge))[[1]], coef.value = matrix(coef(model.ridge)))

 predictions.ridge <- model.ridge%>% predict(x_train)%>% as.vector()
 
 
 ggplot(data, aes(DATE)) + 
   geom_line(aes(y = y_train, colour = "Actual")) + 
   geom_line(aes(y = predictions.ridge, colour = "Predicted"))


data.frame(
  RMSE.r = RMSE(predictions.ridge, y_train),
  Rsquare.r = R2(predictions.ridge, y_train))
  
results<-cbind(data[c("revenue","DATE")],predictions.ridge)


##############################Calculate Response Curves
media_adstock=c("tv_S_adstock", "ooh_S_adstock", "print_S_adstock", "facebook_S_adstock", "search_S_adstock")
media_spend_response_data =  data.frame()


for( i in media_adstock){
df<-filter(coefs , coef.name == i )
  coef_val<-df$coef.value
  
df2<-final_data[[i]]
response=df2*coef_val

spend_response_temp_df<-data.frame(Spend=df2 , Response=response,media_channel=i)

media_spend_response_data<-rbind(media_spend_response_data,spend_response_temp_df)
}


ggplot(media_spend_response_data, aes(x = Spend ,y = Response, color = media_channel)) +
  geom_line(size = 0.8) +
  scale_color_manual(values=c("#005f73", "#0a9396", "#94d2bd","#e9d8a6", "#ee9b00"))
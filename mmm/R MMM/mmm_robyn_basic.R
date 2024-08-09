##Facebooks Robyn MMM
#STEP 1 Load required Packages
#STEP 2 Load data
#STEP 3 Specify input variables
#STEP 4 Specify hyperparameter names and ranges
#STEP 5 Build an initial model
#STEP 6: Select and save any one model
#STEP 7: Get Budget Allocation

################################################################
##                STEP 1 Load required Packages
################################################################
library(Robyn) 
library(reticulate)
set.seed(123)


#Step 1.b Setup virtual Environment & Install nevergrad library
virtualenv_create("r-reticulate")
use_virtualenv("r-reticulate", required = TRUE)
py_install("nevergrad", pip = TRUE)
use_virtualenv("r-reticulate", required = TRUE)

setwd("C:/Users/charl/OneDrive/Desktop/robyn_mmm")


################################################################
##                STEP 2 Load data
################################################################

#load simulated data
data("dt_simulated_weekly")

#Load holidays data from Prophet
data("dt_prophet_holidays")

# Export results to desired directory.
robyn_object<- "C:/Users/charl/OneDrive/Desktop/robyn_mmm"


################################################################
##                STEP 3 Specify input variables
################################################################

InputCollect <- robyn_inputs(
  dt_input = dt_simulated_weekly,
  dt_holidays = dt_prophet_holidays,
  date_var = "DATE", # date format must be "2020-01-01"
  dep_var = "revenue", # there should be only one dependent variable
  dep_var_type = "revenue", # "revenue" (ROI) or "conversion" (CPA)
  prophet_vars = c("trend", "season", "holiday"), # "trend","season", "weekday" & "holiday"
  prophet_country = "DE", # input one country. dt_prophet_holidays includes 59 countries by default
  context_vars = c("competitor_sales_B", "events"), # e.g. competitors, discount, unemployment etc
  paid_media_spends = c("tv_S", "ooh_S", "print_S", "facebook_S", "search_S"), # mandatory input
  paid_media_vars = c("tv_S", "ooh_S", "print_S", "facebook_I", "search_clicks_P"), # mandatory.
  organic_vars = "newsletter", # marketing activity without media spend
  # factor_vars = c("events"), # force variables in context_vars or organic_vars to be categorical
  window_start = "2016-11-21",
  window_end = "2018-08-20",
  adstock = "geometric" # geometric, weibull_cdf or weibull_pdf.
)
print(InputCollect)

################################################################
##                STEP 4 Specify hyperparameter names and ranges
################################################################

#get names of hyperparameters
hyper_names(adstock = InputCollect$adstock, all_media = InputCollect$all_media)

#adstock & saturation hyperparameters

plot_adstock(plot = TRUE)
plot_saturation(plot = TRUE)

# To check maximum lower and upper bounds
hyper_limits()

##  Set individual hyperparameter bounds. They either contain two values e.g. c(0, 0.5)
hyperparameters <- list(
  facebook_S_alphas = c(0.5, 3),
  facebook_S_gammas = c(0.3, 1),
  facebook_S_thetas = c(0, 0.3),
  print_S_alphas = c(0.5, 3),
  print_S_gammas = c(0.3, 1),
  print_S_thetas = c(0.1, 0.4),
  tv_S_alphas = c(0.5, 3),
  tv_S_gammas = c(0.3, 1),
  tv_S_thetas = c(0.3, 0.8),
  search_S_alphas = c(0.5, 3),
  search_S_gammas = c(0.3, 1),
  search_S_thetas = c(0, 0.3),
  ooh_S_alphas = c(0.5, 3),
  ooh_S_gammas = c(0.3, 1),
  ooh_S_thetas = c(0.1, 0.4),
  newsletter_alphas = c(0.5, 3),
  newsletter_gammas = c(0.3, 1),
  newsletter_thetas = c(0.1, 0.4)
)

#add hyperparameters into robyn_inputs()

InputCollect <- robyn_inputs(InputCollect = InputCollect, hyperparameters = hyperparameters)
print(InputCollect)

################################################################
##                STEP 5 Build an initial model
################################################################

OutputModels <- robyn_run(
  InputCollect = InputCollect, # feed in all model specification
  cores = NULL, # NULL defaults to max available - 1
  iterations = 2000, # 2000 recommended for the dummy dataset with no calibration
  trials = 5, # 5 recommended for the dummy dataset
  add_penalty_factor = FALSE, # Experimental feature. Use with caution.
  outputs = FALSE # outputs = FALSE disables direct model output - robyn_outputs()
)
print(OutputModels)

## Calculate Pareto fronts, cluster and export results and plots.

OutputCollect <- robyn_outputs(
  InputCollect, OutputModels,
  csv_out = "pareto",
  pareto_fronts = "auto",
  clusters = TRUE,
  export = TRUE,
  plot_pareto = TRUE,
  plot_folder = robyn_object
  
)
print(OutputCollect)

################################################################
##                STEP 6: Select and save any one model
################################################################

## Compare all model one-pagers and select one that largely reflects your business reality.
print(OutputCollect)
select_model <- "4_279_7"

ExportedModel <- robyn_write(InputCollect, OutputCollect, select_model, export = TRUE)
print(ExportedModel)

################################################################
##                STEP 7: Get Budget Allocation 
################################################################

#Get budget allocation based on the selected model above

# Check media summary for selected model
print(ExportedModel)

# NOTE: The order of constraints should follow:
InputCollect$paid_media_spends

# Run the "max_historical_response" scenario: "What's the revenue lift potential with the
# same historical spend level and what is the spend mix?"
AllocatorCollect1 <- robyn_allocator(
  InputCollect = InputCollect,
  OutputCollect = OutputCollect,
  select_model = select_model,
  scenario = "max_historical_response",
  channel_constr_low = 0.7,
  channel_constr_up = c(1.2, 1.5, 1.5, 1.5, 1.5),
  export = TRUE,
  date_min = "2016-11-21",
  date_max = "2018-08-20"
)
print(AllocatorCollect1)
# plot(AllocatorCollect1).


Spend1 <- 20000
################################################################
##                STEP 8: Get Response Curves
################################################################

Response <- robyn_response(
    json_file = "C:/Users/charl/OneDrive/Desktop/robyn_mmm/Robyn_202405230911_init/RobynModel-4_279_7.json",
     dt_input = dt_simulated_weekly,
     dt_holidays = dt_prophet_holidays,
     metric_name = "facebook_S",
    metric_value = Spend1
   )
Response$plot


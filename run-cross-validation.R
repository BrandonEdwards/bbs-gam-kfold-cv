#######################################
# Adam C. Smith & Brandon P.M. Edwards
# GAM k-Fold Cross Validation
# run-cross-validation.R
# Created May 2018
# Last Updated May 2018
#######################################

#######################################
# Clear Memory
#######################################

remove(list = ls())

#######################################
# Set Constants
#######################################

bbsDataPath <- "input/bbs data 2016 continent.RData"
#speciesFilePath <- "input/AIspeciesToTest.csv"
adaptSteps = 500  # Number of steps to "tune" the samplers.
burnInSteps = 20000 # Number of steps to "burn-in" the samplers.
nChains = 3 # Number of chains to run.
numSavedSteps=2000 # Total number of steps to save.
thinSteps=10 # Number of steps to "thin" (1=keep every step).
nIter = ceiling( ( numSavedSteps * thinSteps ) / nChains ) # Steps per chain.
nCores <- 3
speciesToTest <- "Barn Swallow"

#######################################
# Import Libraries and Files
#######################################


#install.packages("jagsUI")
#install.packages("foreach")
#install.packages("doParallel")
library(jagsUI)
library(foreach)
library(doParallel)

source("src/data-prep-functions.r")
source("src/jags-functions.R")

dir.create("output")

#######################################
# Read Data
#######################################

load(bbsDataPath)
mod <- ("models/GAM.txt")  
looMod <- ("models/GAM-LOOCV.txt")
#speciesToTest <- read.csv(speciesFilePath, header = F)$V1

#######################################
# Wrangle Data
#######################################

data.cleaned <- cleanData(birds,
                          datacount.sp,
                          route,
                          sp,
                          st.areas)

speciesIndex <- getSpeciesIndex(data.cleaned$species,
                                speciesToTest) #change later to list

spNum <- 1
totalSp <- length(speciesToTest)
for (index in speciesIndex)
{
  print(paste("Sp. ",spNum, "/",totalSp))
  data.prep <- speciesDataPrep(bbsDataPath,
                               data.cleaned$species,
                               data.cleaned$unmod.sp,
                               data.cleaned$sptorun,
                               data.cleaned$sptorun2,
                               index)
  
#######################################
# Full Model Run
#######################################
  
  data.jags <- list(nknots = data.prep$nknots,
                    X.basis = data.prep$X.basis,
                    ncounts = nrow(data.prep$spsp.f), 
                    nstrata=length(unique(data.prep$spsp.f$strat)), 
                    ymin = data.prep$ymin, 
                    ymax = data.prep$ymax,
                    nonzeroweight = data.prep$pR.wts$p.r.ever, 
                    count = as.integer(data.prep$spsp.f$count), 
                    strat = as.integer(data.prep$spsp.f$strat), 
                    obser = as.integer(data.prep$spsp.f$obser), 
                    year = data.prep$spsp.f$year,
                    firstyr = data.prep$spsp.f$firstyr,
                    nobservers = data.prep$nobservers,
                    fixedyear = midyear)
  
  sp.params = c("beta.X",
                "strata",
                "STRATA",
                "n")
  
  print(paste("Sp. ",spNum, "/",totalSp, " ", data.prep$sp.1, " FULL RUN ", date(), sep = ""))
  jagsModFull <- runModel(data.jags,
                          NULL,
                          sp.params,
                          mod,
                          nChains,
                          adaptSteps,
                          nIter,
                          burnInSteps,
                          thinSteps,
                          parallel = TRUE)
  
  # Save the entire jags file for future use
  save(jagsModFull, file = paste(data.prep$dir, "/full.Rdata", sep=""))
  
  # Extract the coefficients for each parameter to use as initializations 
  mcmc.paramsC1 <- as.list(jagsModFull$model$cluster1$state()[[1]])
  mcmc.paramsC2 <- as.list(jagsModFull$model$cluster2$state()[[1]])
  mcmc.paramsC3 <- as.list(jagsModFull$model$cluster3$state()[[1]])
  
  mcmc.params <- list(mcmc.paramsC1, mcmc.paramsC2, mcmc.paramsC3)
  
#######################################
# k-Fold Cross validation
#######################################
  
  #Set up parallelization stuff
  cluster <- makeCluster(nCores, type = "PSOCK")
  registerDoParallel(cluster)
  
  foreach(year=data.prep$ymin:data.prep$ymax, .packages = 'jagsUI') %dopar%
  {
    indicesToRemove <- which(data.prep$spsp.f$year == year)
    trueCount <- data.prep$spsp.f[indicesToRemove, ]$count
    
    nRemove <- as.integer(length(indicesToRemove))
    
    temp <- data.prep$spsp.f
    temp[indicesToRemove, ]$count <- NA
    
    data.jags <- list(nknots = data.prep$nknots,
                      X.basis = data.prep$X.basis,
                      ncounts = nrow(temp), 
                      nstrata=length(unique(temp$strat)), 
                      ymin = data.prep$ymin, 
                      ymax = data.prep$ymax,
                      nonzeroweight = data.prep$pR.wts$p.r.ever, 
                      count = as.integer(temp$count), 
                      strat = as.integer(temp$strat), 
                      obser = as.integer(temp$obser), 
                      year = temp$year,
                      firstyr = temp$firstyr,
                      nobservers = data.prep$nobservers,
                      fixedyear = midyear,
                      I = indicesToRemove,
                      Y = trueCount,
                      nRemove = nRemove)
    
    params <- c("logprob", "LambdaSubset")
    
    print(paste("Sp. ",spNum, "/",totalSp, " ", data.prep$sp.1, " Year ", year, "/", 
                data.prep$ymax, " removed ", date(), sep = ""))
    # re-run the model with the new dataset (same data as before, just with NAs this time)
    jagsjob = runModel(data.jags, mcmc.params, params, looMod,
                       nChains = 3, adaptSteps, nIter, 0, 
                       thinSteps, parallel = FALSE)
    
    save(jagsjob, file = paste(data.prep$dir, "/year", year, 
                               "removed.Rdata", sep=""))
  }
  
  stopCluster(cluster)
  
  spNum <- spNum + 1
}


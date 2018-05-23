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
speciesToTest <- c("King Rail")
#setwd("C:/2016gam")

#######################################
# Import Libraries and Files
#######################################

#install.packages("runjags")
#install.packages("rjags")
#install.packages("R2jags")
library(runjags)
library(rjags)
library(R2jags)

source("src/data-prep-functions.r")

#######################################
# Read Data
#######################################

load(bbsDataPath)
#mod <- ("models/bbs model 16 hierarchical GAM rescaled heavy tails.txt")  
#looMod <- ("models/bbs model 16 hierarchical GAM rescaled heavy tails LOOCV.txt") 

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

data.prep <- speciesDataPrep(bbsDataPath,
                             data.cleaned$species,
                             data.cleaned$unmod.sp,
                             data.cleaned$sptorun,
                             data.cleaned$sptorun2,
                             speciesIndex)

#######################################
# JAGS Setup
#######################################

data.jags <- list(nknots = nknots,
                  X.basis = X.basis,
                  ncounts = nrow(spsp.f), 
                  nstrata=length(unique(spsp.f$strat)), 
                  ymin = ymin, 
                  ymax = ymax, 
                  #yminsc = yminsc, 
                  #ymaxsc = ymaxsc, 
                  #yminpred = yminpred, 
                  #ymaxpred = ymaxpred, 
                  #yearscale = spsp.f$yearscale,
                  nonzeroweight = pR.wts$p.r.ever, 
                  count = as.integer(spsp.f$count), 
                  strat = as.integer(spsp.f$strat), 
                  obser = as.integer(spsp.f$obser), 
                  year = spsp.f$year,
                  firstyr = spsp.f$firstyr, 
                  #fixedyear = midyear2, 
                  nobservers = nobservers)

sp.params = c("beta.X",
              "B.X",
              "tauX",
              "sdbeta",
              "strata",
              "STRATA",
              #"sdstrata",
              "sdobs",
              #"obs",
              "n",
              "posdiff",
              "sdnoise",
              "eta",
              "overdisp",
              "taulogtauobs",
              "mulogtauobs",
              # "exmaxlambda",
              # "exminlambda",
              # "minlambda",
              # "maxlambda",
              "maxf",
              "meanf",
              "nfzero",
              "gof",
              "fgof",
              "diffgof")

if("jagsMod" %in% ls()){
  rm(list= ls()[which(ls() %in% c("jagsMod"))])
}

adaptSteps = 500  # Number of steps to "tune" the samplers.
burnInSteps = 20000 # Number of steps to "burn-in" the samplers.
nChains = 1 # Number of chains to run.
numSavedSteps=2000 # Total number of steps to save.
thinSteps=10 # Number of steps to "thin" (1=keep every step).
nIter = ceiling( ( numSavedSteps * thinSteps ) / nChains ) # Steps per chain.

t1 = Sys.time()

#######################################
# JAGS Initialization and Burn-in
#######################################

# Initialization
jagsMod = jags.model( mod, 
                      data= data.jags ,  
                      #inits= sp.inits,  
                      n.chains= nChains , 
                      n.adapt= 0 )
adaptest <- F
while(adaptest == F)
{
  adaptest <- adapt(jagsMod,n.iter = adaptSteps)
}

# Burn-in
cat( "Burning in the MCMC chain...\n" )
update( jagsMod , n.iter=burnInSteps )

# Extract the coefficients for each parameter to use as initializations 
mcmc.params <- coef(jagsMod)

#######################################
# Leave one Year Out CV
#######################################
year <- 1
#for (year in ymin:ymax)
#{
  indicesToRemove <- which(spsp.f$year == year)
  trueCount <- spsp.f[indicesToRemove, ]$count
  I <- indicesToRemove
  Y <- trueCount
  nRemove <- as.integer(length(I))
  
  temp <- spsp.f
  temp[indicesToRemove, ]$count <- NA
  
  inits=function(){list(B.X = mcmc.params$B.X,
                        Elambda = mcmc.params$Elambda,
                        STRATA = mcmc.params$STRATA,
                        beta.X = mcmc.params$beta.X,
                        eta = mcmc.params$eta,
                        fcount = mcmc.params$fcount,
                        logtauobs = mcmc.params$logtauobs,
                        mulogtauobs = mcmc.params$mulogtauobs,
                        nu = mcmc.params$nu,
                        obs = mcmc.params$obs,
                        sdbeta = mcmc.params$sdbeta,
                        strata = mcmc.params$strata,
                        tauX = mcmc.params$tauX,
                        taulogtauobs = mcmc.params$taulogtauobs,
                        taunoise = mcmc.params$taunoise,
                        taustrata = mcmc.params$taustrata,
                        tauyear = mcmc.params$tauyear)}
  
  data.jags <- list(nknots = nknots,
                    X.basis = X.basis,
                    ncounts = nrow(temp), 
                    nstrata=length(unique(temp$strat)), 
                    ymin = ymin, 
                    ymax = ymax, 
                    #yminsc = yminsc, 
                    #ymaxsc = ymaxsc, 
                    #yminpred = yminpred, 
                    #ymaxpred = ymaxpred, 
                    #yearscale = spsp.f$yearscale,
                    nonzeroweight = pR.wts$p.r.ever, 
                    count = as.integer(temp$count), 
                    strat = as.integer(temp$strat), 
                    obser = as.integer(temp$obser), 
                    year = temp$year,
                    firstyr = temp$firstyr, 
                    #fixedyear = midyear2, 
                    nobservers = nobservers,
                    I = I,
                    nRemove = nRemove)
  
  params <- c("logprob", "LambdaSubset")
  
  # re-run the model with the new dataset (same data as before, just with NAs this time)
  jagsjob = jags(model = looMod,
                 inits = inits,
                 data = data.jags,
                 param = params,
                 n.chain = 1,
                 n.burnin = 0,
                 n.iter = nIter,
                 n.thin = 1)
  # use these values to calculate the cross validation statistic for left out counts  
  
  monitoredValues <- as.data.frame(jagsjob$BUGSoutput$mean)
  
  #logprob <- NULL
  #for (i in 1:nRemove)
  #{
  #  tempProb <- ((-1)*estimates$lambda[I[i]]) + (trueCount[i]*estimates$Elambda[I[i]])-log(factorial(trueCount[i]))
  #  logprob <- c(logprob, tempProb)
  #}
  
#  lambdaEstimates <- estimates[indicesToRemove, ]$lambda
  
 # nRemove/nrow(spsp.f) * (sum((trueCount-lambdaEstimates)^2)/nRemove)
#}


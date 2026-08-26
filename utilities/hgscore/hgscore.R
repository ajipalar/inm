options(repos = c(CRAN = "https://cloud.r-project.org/"))
setwd(".")
##if (!require("BiocManager", quietly = TRUE))
##	    install.packages("BiocManager")
##BiocManager::install("SMAD")

##browseVignettes("SMAD")
install.packages("optparse")
install.packages("tidyverse")

library("readr")
library("SMAD")
library("optparse")
##HG(TestDatInput)

data <- read.csv("cullin_hgscore_input.csv")

output <- HG(data)

write_csv(output, "hgscore_output.csv")

dplyr::last_dplyr_warnings()


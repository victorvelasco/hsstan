setwd("hsstan/")
install.packages("devtools")
devtools::load_all()



set.seed(1)

D <- 7350
n <- 817

data_train <- sapply(1:D, function(i) rnorm(n, 0, 0.5))
y <- rep(0, n)
linpred <- 0.5 +  data_train[,1:6] %*% c(-2.5,-1.5,-0.5,0.5,1.5,2.5) + data_train[,11:16] %*% c(-2.5,-1.5,-0.5,0.5,1.5,2.5)
sample(0:1, size = 1, prob = )

data_train <- cbind(sample(c(0, 1), size = n, replace = TRUE, prob = c(0.36, 0.64)), data_train)
colnames(data_train) <- c("y", paste0("x", 1:D))
data_train <- as_tibble(data_train)

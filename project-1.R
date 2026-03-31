# Loading Required Libraries
required_packages <- c("tidyverse", "stringr", "caret", "ggplot2", "corrplot", 
                       "GGally", "glmnet", "rpart", "rpart.plot", "gains", "car")

new_packages <- required_packages[!(required_packages %in% installed.packages())]
if(length(new_packages)) install.packages(new_packages)

lapply(required_packages, library, character.only = TRUE)

# Data Loading and Preparation
used_cars_df <- read.csv(file.choose())

cols_to_drop <- intersect(c("New_Price", "Name"), colnames(used_cars_df))
used_cars_df <- used_cars_df %>% select(-all_of(cols_to_drop))

used_cars_df <- used_cars_df %>%
  mutate(
    Mileage = as.numeric(str_extract(Mileage, "\\d+\\.*\\d*")),
    Engine  = as.numeric(str_extract(Engine, "\\d+\\.*\\d*")),
    Power   = as.numeric(str_extract(Power, "\\d+\\.*\\d*"))
  ) %>%
  drop_na(Price)

num_cols <- used_cars_df %>% select(where(is.numeric)) %>% names()
used_cars_df[num_cols] <- lapply(used_cars_df[num_cols], function(x) ifelse(is.na(x), median(x, na.rm = TRUE), x))

categorical_cols <- intersect(c("Location", "Fuel_Type", "Transmission", "Owner_Type"), names(used_cars_df))
used_cars_df <- used_cars_df %>% mutate(across(all_of(categorical_cols), as.factor))

write.csv(used_cars_df, "clean_used_cars.csv", row.names = FALSE)

# Exploratory Analysis
summary(select(used_cars_df, where(is.numeric)))
cor_matrix <- cor(select(used_cars_df, where(is.numeric)))
corrplot(cor_matrix, method = "color", type = "upper", tl.cex = 0.7)

ggplot(used_cars_df, aes(x = Price)) + geom_histogram(bins = 40, fill = "skyblue", color = "black") + theme_minimal()
ggplot(used_cars_df, aes(x = Fuel_Type, y = Price)) + geom_boxplot(fill = "lightblue") + theme_minimal()
ggplot(used_cars_df, aes(x = Transmission, y = Price)) + geom_boxplot(fill = "salmon") + theme_minimal()
ggplot(used_cars_df, aes(x = Owner_Type, y = Price)) + geom_boxplot(fill = "lightgreen") + theme_minimal()

ggplot(used_cars_df, aes(x = Kilometers_Driven, y = Price)) + geom_point(alpha = 0.4) + theme_minimal()
ggplot(used_cars_df, aes(x = Engine, y = Price)) + geom_point(alpha = 0.4) + theme_minimal()
ggplot(used_cars_df, aes(x = Power, y = Price)) + geom_point(alpha = 0.4) + theme_minimal()
ggplot(used_cars_df, aes(x = Mileage, y = Price)) + geom_point(alpha = 0.4) + theme_minimal()

# Data Partitioning
set.seed(42)
trainIndex <- createDataPartition(used_cars_df$Price, p = 0.7, list = FALSE)
train_df <- used_cars_df[trainIndex, ]
test_df  <- used_cars_df[-trainIndex, ]

# Model Building

mlr_model <- train(Price ~ ., data = train_df, method = "lm")
mlr_pred <- predict(mlr_model, test_df)
mlr_rmse <- RMSE(mlr_pred, test_df$Price)
mlr_r2   <- R2(mlr_pred, test_df$Price)

vif_values <- vif(mlr_model$finalModel)

pre_proc <- preProcess(train_df, method = c("center", "scale"))
train_knn <- predict(pre_proc, train_df)
test_knn  <- predict(pre_proc, test_df)
knn_model <- train(Price ~ ., data = train_knn, method = "knn", tuneLength = 10)
knn_pred <- predict(knn_model, test_knn)
knn_rmse <- RMSE(knn_pred, test_knn$Price)
knn_r2   <- R2(knn_pred, test_knn$Price)

cart_model <- rpart(Price ~ ., data = train_df, method = "anova")
rpart.plot(cart_model)
cart_pred <- predict(cart_model, test_df)
cart_rmse <- RMSE(cart_pred, test_df$Price)
cart_r2   <- R2(cart_pred, test_df$Price)

x_train <- model.matrix(Price ~ ., data = train_df)[, -1]
y_train <- train_df$Price
x_test  <- model.matrix(Price ~ ., data = test_df)[, -1]

ridge_model <- cv.glmnet(x_train, y_train, alpha = 0)
ridge_pred  <- predict(ridge_model, s = ridge_model$lambda.min, newx = x_test)
ridge_rmse  <- RMSE(ridge_pred, test_df$Price)
ridge_r2    <- R2(ridge_pred, test_df$Price)

lasso_model <- cv.glmnet(x_train, y_train, alpha = 1)
lasso_pred  <- predict(lasso_model, s = lasso_model$lambda.min, newx = x_test)
lasso_rmse  <- RMSE(lasso_pred, test_df$Price)
lasso_r2    <- R2(lasso_pred, test_df$Price)

results <- data.frame(
  Model = c("Linear Regression", "k-NN", "CART", "Ridge", "Lasso"),
  RMSE  = c(mlr_rmse, knn_rmse, cart_rmse, ridge_rmse, lasso_rmse),
  R2    = c(mlr_r2, knn_r2, cart_r2, ridge_r2, lasso_r2)
)
print(results)

# Visualization

residuals_mlr <- residuals(mlr_model$finalModel)
fitted_vals_mlr <- fitted(mlr_model$finalModel)
ggplot(NULL, aes(x = fitted_vals_mlr, y = residuals_mlr)) +
  geom_point(alpha = 0.5) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Residuals vs Fitted Values (MLR)", x = "Fitted Values", y = "Residuals") +
  theme_minimal()

lift_df <- data.frame(actual = test_df$Price, predicted = as.vector(ridge_pred)) %>%
  arrange(desc(predicted)) %>%
  mutate(bucket = ntile(predicted, 10))
lift_summary <- lift_df %>%
  group_by(bucket) %>%
  summarise(mean_actual = mean(actual), .groups = 'drop')

ggplot(lift_summary, aes(x = bucket, y = mean_actual)) +
  geom_line(color = "blue") +
  geom_point(color = "darkblue") +
  labs(title = "Lift Chart (Ridge Model)", x = "Prediction Decile", y = "Average Actual Price") +
  theme_minimal()

importance <- varImp(cart_model)
imp_df <- data.frame(Variable = rownames(importance), Importance = importance$Overall)
ggplot(imp_df, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Variable Importance (CART Model)", x = "Variable", y = "Importance Score") +
  theme_minimal()

best_model <- results %>% arrange(RMSE) %>% slice(1)
cat("✅ Best performing model based on RMSE is:", best_model$Model, "\n")

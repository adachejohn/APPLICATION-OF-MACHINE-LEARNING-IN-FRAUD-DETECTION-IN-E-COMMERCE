

##EKERE JOHN ADACHE
##ID 2123522
## TOPIC: APPLICATION OF MACHINE LEARNING IN FRAUD DETECTION IN E-COMMERCE; A COMPARATIVE TEST BETWEEN FOUR ML ALGORITHMS

# Load necessary libraries
install.packages(c("dplyr", "caret", "randomForest", "neuralnet", "ROCR", "outliers"))
library(dplyr)
library(caret)
library(randomForest)
library(neuralnet)
library(ROCR)
library(outliers)

# Load the dataset
setwd("C:\\Users\\localuser\\OneDrive - University of Bolton\\Documents")
Data <- read.csv("ecommerce.csv")

# Step 1: Data Preprocessing
# Check for missing values
missing_values <- colSums(is.na(Data))
print(missing_values[missing_values > 0])

# Normalize the 'Amount' column
Data$Amount <- scale(Data$Amount)

# Step 2: Exploratory Data Analysis (EDA)
table(Data$Class)
plot(density(Data$Amount[Data$Class == 0]), main="Amount Distribution (Non-Fraud)")
lines(density(Data$Amount[Data$Class == 1]), col="red")
legend("topright", legend=c("Non-Fraud", "Fraud"), col=c("black", "red"), lty=1)

# Step 3: Feature Engineering (Adding a 'Time_Since_Last' Feature)
Data <- Data[order(Data$Time), ]
Data$Time_Since_Last <- c(NA, diff(Data$Time))
max_time_diff <- max(Data$Time_Since_Last, na.rm = TRUE)
Data$Time_Since_Last[is.na(Data$Time_Since_Last)] <- max_time_diff

# Step 4: Data Splitting
set.seed(123)
train_index <- createDataPartition(Data$Class, p = 0.7, list = FALSE)
train_data <- Data[train_index, ]
test_data <- Data[-train_index, ]

# Identify rows tagged as fraud
fraud_rows <- Data[Data$Class == 1, ]

# Print the number of rows tagged as fraud
num_fraud_rows <- nrow(fraud_rows)
cat("Number of rows tagged as fraud:", num_fraud_rows, "\n")

# Step 5: Model Selection (Random Forest, Neural Network, Isolation Forest, and Logistic Regression)

# Step 6: Model Training
# Random Forest
rf_model <- randomForest(Class ~ ., data = train_data, ntree = 100)

# Neural Network
nn_model <- neuralnet(Class ~ ., data = train_data, hidden = c(5, 2))

# Isolation Forest
if_model <- dbscan::dbscan(train_data[, -ncol(train_data)], eps = 0.5, minPts = 5)

# Logistic Regression
lr_model <- glm(Class ~ ., data = train_data, family = "binomial")

# Step 7: Model Evaluation
# Random Forest
rf_predictions <- predict(rf_model, newdata = test_data)
rf_predictions <- as.factor(ifelse(rf_predictions == 1, "Fraud", "Non-Fraud"))

# Neural Network
nn_predictions <- predict(nn_model, newdata = test_data)
nn_predictions <- as.factor(ifelse(nn_predictions > 0.5, "Fraud", "Non-Fraud"))

# Isolation Forest
if_predictions <- ifelse(if_model$cluster == 0, "Normal", "Anomaly")

# Logistic Regression
lr_predictions <- predict(lr_model, newdata = test_data, type = "response")
lr_predictions <- as.factor(ifelse(lr_predictions > 0.5, "Fraud", "Non-Fraud"))

# Step 8: Cross-Validation (Implementing k-fold cross-validation)
# Define the number of folds
num_folds <- 5

# Create data partition object for cross-validation
folds <- createFolds(Data$Class, k = num_folds)

# Initialize a list to store evaluation results for each fold
fold_results <- list()

# Perform k-fold cross-validation
for (i in 1:num_folds) {
  train_indices <- unlist(folds[i])
  test_indices <- setdiff(1:nrow(Data), train_indices)
  train_data <- Data[train_indices, ]
  test_data <- Data[test_indices, ]
  
  # Step 9: Statistical Anomaly Detection (Z-Score) and Grubbs' Test
  # Apply Grubbs' test to detect outliers in the 'Amount' variable
  grubbs_test_result <- grubbs.test(Data$Amount, type = 10, opposite = TRUE)
  identified_outliers <- grubbs_test_result$outliers
  cat("Identified outliers using Grubbs' test:")
  print(identified_outliers)
  
  # Calculate the Z-score for the 'amount' variable
  mean_amount <- mean(Data$Amount)
  sd_amount <- sd(Data$Amount)
  Data$Z_Score <- abs((Data$Amount - mean_amount) / sd_amount)
  z_score_threshold <- 3
  anomalies_zscore <- Data$Z_Score > z_score_threshold
  cat("Number of anomalies detected using Z-Score:", sum(anomalies_zscore), "\n")
  
  # Step 10: Model Tuning (Random Forest)
  rf_grid <- expand.grid(
    mtry = c(2, 4, 6),        # Number of variables randomly chosen at each split
    nodesize = c(1, 5, 10)    # Minimum size of terminal nodes
  )
  ctrl <- trainControl(method = "cv", number = 5)  # 5-fold cross-validation
  set.seed(123)  # Set a seed for reproducibility
  rf_tuned <- train(Class ~ ., data = train_data, method = "rf",
                    trControl = ctrl, tuneGrid = rf_grid)
  best_mtry <- rf_tuned$bestTune$mtry
  best_nodesize <- rf_tuned$bestTune$nodesize
  final_rf_model <- randomForest(Class ~ ., data = train_data,
                                 mtry = best_mtry, nodesize = best_nodesize)
  final_rf_predictions <- predict(final_rf_model, newdata = test_data)
  
  # Step 11: Anomaly Detection (Using the trained models)
  rf_anomalies <- ifelse(rf_predictions == "Fraud", "Anomaly", "Normal")
  nn_anomalies <- ifelse(nn_predictions == "Fraud", "Anomaly", "Normal")
  
  # Step 12: Visualization (Visualize anomalies detected by each model)
  create_anomaly_scatter_plot <- function(data, model_name) {
    ggplot(data, aes(x = Amount, color = factor(Class))) +
      geom_point(aes_string(shape = paste(model_name, "Anomaly")), alpha = 0.7) +
      labs(title = paste("Anomalies Detected by", model_name, "Model"),
           x = "Amount", y = "Class",
           color = "Actual Fraud",
           shape = paste(model_name, "Anomaly")) +
      scale_color_manual(values = c("blue", "red"))
  }
  
  test_data[[paste("Random_Forest", "Anomaly")]] <- as.factor(rf_anomalies)
  plot_name <- paste("Anomalies_Detected_by_", "Random_Forest", "_Model.png")
  ggsave(plot = create_anomaly_scatter_plot(test_data, "Random_Forest"),
         filename = plot_name,
         width = 8, height = 6)
  print(paste("Saved plot:", plot_name))
  
  test_data[[paste("Neural_Network", "Anomaly")]] <- as.factor(nn_anomalies)
  plot_name <- paste("Anomalies_Detected_by_", "Neural_Network", "_Model.png")
  ggsave(plot = create_anomaly_scatter_plot(test_data, "Neural_Network"),
         filename = plot_name,
         width = 8, height = 6)
  print(paste("Saved plot:", plot_name))
  
  # Step 13: Model Comparison
  calculate_metrics <- function(predictions, actual) {
    confusion <- confusionMatrix(predictions, actual)
    accuracy <- confusion$overall["Accuracy"]
    precision <- confusion$byClass["Pos Pred Value"]
    recall <- confusion$byClass["Sensitivity"]
    f1_score <- confusion$byClass["F1"]
    auc <- roc(actual, as.numeric(predictions))$auc
    return(list(Accuracy = accuracy, Precision = precision, Recall = recall, F1_Score = f1_score, AUC = auc))
  }
  
  models <- list(
    Random_Forest = rf_predictions,
    Neural_Network = nn_predictions
  )
  
  metrics <- lapply(models, function(predictions) {
    calculate_metrics(predictions, test_data$Class)
  })
  
  metrics_df <- do.call(rbind, metrics)
  rownames(metrics_df) <- names(models)
  
  print("Model Comparison Metrics:")
  print(metrics_df)
  
  # Step 14: Deployment (Not covered in this analysis)
  # Step 15: Monitoring and Maintenance (Not covered in this analysis)
  # End of loop for k-fold cross-validation
}

# Additional Analysis
# Step 16: Statistical Tests for Anomaly Detection (Z-Score and Grubbs' Test)
# Step 17: Checking for Missing Values (Already covered)
# Step 18: Identifying Columns Tagged as Fraud
fraud_columns <- Data[, Data$Class == 1]

# Step 19: Showing a Table of Only Rows Tagged as Fraud
fraud_rows <- Data[Data$Class == 1, ]
View(fraud_rows)

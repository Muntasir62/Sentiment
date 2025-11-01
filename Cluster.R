install.packages("proxy")
install.packages("factoextra")
install.packages("dbscan")
library(tm)
library(dplyr)
library(SnowballC)
library(proxy)  
library(ggplot2)  
library(factoextra)  
library(dbscan)


hotel_reviews <- read.csv("D:/Data Science/Final/Hotel_Reviews.csv")

head(hotel_reviews)
#  Combine Positive and Negative Reviews
hotel_reviews$combined_review <- paste(hotel_reviews$Positive_Review, hotel_reviews$Negative_Review)

#  Create a Small Sample

hotel_reviews_sample <- sample_n(hotel_reviews, 10000)

#  Create a Corpus from the Sample
corpus <- VCorpus(VectorSource(hotel_reviews_sample$combined_review))

#  Apply ALL Text Processing Steps

corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, stemDocument) 
# Create dtm with tf-idf weighting
# Create DTM with term frequency first
dtm <- DocumentTermMatrix(corpus)

# Then apply TF-IDF transformation manually
dtm_tfidf <- weightTfIdf(dtm)

# Clean the TF-IDF DTM by removing sparse terms
dtm_tfidf_cleaned <- removeSparseTerms(dtm_tfidf, sparse = 0.99)

# View the results
print("Dimensions of the cleaned TF-IDF DTM:")
print(dim(dtm_tfidf_cleaned))

print("First 5 rows and columns of the cleaned TF-IDF DTM:")
inspect(dtm_tfidf_cleaned[1:5, 1:5])
inspect(dtm_tfidf_cleaned[1:5, 6:10])

# print most frequent terms

findFreqTerms(dtm_tfidf, lowfreq = 100)

# Verify the weighting method
print("Weighting method:")
print(dtm_tfidf_cleaned$weighting)

# Convert to matrix for clustering
tfidf_matrix <- as.matrix(dtm_tfidf_cleaned)
# Remove any documents with all zeros
nonzero_rows <- which(rowSums(tfidf_matrix) > 0)
tfidf_matrix <- tfidf_matrix[nonzero_rows, ]
print(paste("Working with", nrow(tfidf_matrix), "non-zero documents"))

# K-MEANS CLUSTERING
# Determine optimal number of clusters using elbow method
# Use a smaller sample for k-means to make it faster
set.seed(123)
sample_size_km <- min(2000, nrow(tfidf_matrix))  
sample_indices_km <- sample(1:nrow(tfidf_matrix), sample_size_km)
tfidf_sample_km <- tfidf_matrix[sample_indices_km, ]

print(paste("Using sample of", sample_size_km, "documents for k-means clustering"))

#   k-means with fewer iterations
# Calculate within-cluster sum of squares for different k values
print("Calculating within-cluster sum of squares for different k values...")
wss <- sapply(2:10, function(k) {
  kmeans(tfidf_sample_km, centers = k, nstart = 5, iter.max = 10)$tot.withinss
})

# Plot elbow method
plot(2:10, wss, type = "b", pch = 19, frame = FALSE, 
     xlab = "Number of clusters K", ylab = "Total within-clusters sum of squares",
     main = "Elbow Method for Optimal K")

# Find the elbow point (where the decrease in WSS starts to level off)
# Simple method: find where the second derivative is maximized
d1 <- diff(wss)
d2 <- diff(d1)
elbow_point <- which.max(abs(d2)) + 2  # +2 because we started at k=2

cat(paste("\nSuggested optimal k from elbow method:", elbow_point, "\n"))
cat("Review the elbow plot and choose the appropriate k value.\n")

# Use the suggested elbow point
k <- elbow_point

print(paste("Performing K-means clustering with k =", k))

# Perform K-means clustering with the chosen k
kmeans_result <- kmeans(tfidf_sample_km, centers = k, nstart = 5, iter.max = 10)


# View cluster sizes
cat("\nK-MEANS CLUSTERING RESULTS:\n")
print(table(kmeans_result$cluster))

# Get top terms for each cluster
get_top_terms <- function(cluster_centers, n_terms = 10) {
  top_terms <- list()
  for (i in 1:nrow(cluster_centers)) {
    top_indices <- order(cluster_centers[i, ], decreasing = TRUE)[1:n_terms]
    top_terms[[i]] <- colnames(cluster_centers)[top_indices]
  }
  return(top_terms)
}

top_terms_kmeans <- get_top_terms(kmeans_result$centers)
cat("\nTop terms for each K-means cluster:\n")
for (i in 1:length(top_terms_kmeans)) {
  cat(paste("Cluster", i, ":", paste(top_terms_kmeans[[i]], collapse = ", "), "\n"))
}



# ---------------------------
# HIERARCHICAL CLUSTERING
# ---------------------------

# Use  smaller sample for hierarchical clustering
set.seed(123)
sample_size_hc <- min(500, nrow(tfidf_matrix))  
sample_indices_hc <- sample(1:nrow(tfidf_matrix), sample_size_hc)
tfidf_sample_hc <- tfidf_matrix[sample_indices_hc, ]

print(paste("Using sample of", sample_size_hc, "documents for hierarchical clustering"))

# Calculate distance matrix
print("Calculating cosine distance matrix...")
cosine_dist <- dist(tfidf_sample_hc, method = "cosine")

# Perform hierarchical clustering
print("Performing hierarchical clustering...")
hclust_result <- hclust(cosine_dist, method = "ward.D2")

# Plot dendrogram
print("Plotting dendrogram...")
plot(hclust_result, main = "Hierarchical Clustering Dendrogram", 
     xlab = "Documents", sub = "", cex = 0.6)

# Cut dendrogram to get clusters
h_clusters <- cutree(hclust_result, k = k)

# View cluster sizes for hierarchical clustering
cat("\nHIERARCHICAL CLUSTERING RESULTS:\n")
print(table(h_clusters))
# ---------------------------
# DBSCAN CLUSTERING
# ---------------------------
# Use a smaller sample for DBSCAN 
set.seed(123)
sample_size_dbscan <- min(1000, nrow(tfidf_matrix))
sample_indices_dbscan <- sample(1:nrow(tfidf_matrix), sample_size_dbscan)
tfidf_sample_dbscan <- tfidf_matrix[sample_indices_dbscan, ]

print(paste("Using sample of", sample_size_dbscan, "documents for DBSCAN clustering"))

# Calculate cosine distance matrix for DBSCAN
print("Calculating cosine distance for DBSCAN...")
cosine_dist_dbscan <- dist(tfidf_sample_dbscan, method = "cosine")

# Determine optimal epsilon using k-NN distance plot
print("Finding optimal epsilon using k-NN distance...")
# Try multiple epsilon values to find better clustering
print("Testing multiple epsilon values...")

# Function to evaluate DBSCAN with different parameters
test_dbscan_parameters <- function(eps_values, min_pts_values) {
  results <- list()
  for (eps in eps_values) {
    for (min_pts in min_pts_values) {
      dbscan_test <- dbscan(cosine_dist_dbscan, eps = eps, minPts = min_pts)
      clusters <- dbscan_test$cluster
      n_clusters <- length(unique(clusters[clusters != 0]))  
      noise_percent <- sum(clusters == 0) / length(clusters) * 100
      
      results[[paste0("eps_", eps, "_minpts_", min_pts)]] <- list(
        eps = eps,
        min_pts = min_pts,
        n_clusters = n_clusters,
        noise_percent = noise_percent,
        cluster_sizes = table(clusters)
      )
    }
  }
  return(results)
}
# Test different parameter combinations
eps_values <- c(0.1, 0.3, 0.5, 0.7, 0.9)
min_pts_values <- c(3, 5, 7)

param_results <- test_dbscan_parameters(eps_values, min_pts_values)

# Display parameter test results
cat("\nDBSCAN Parameter Test Results:\n")
for (result_name in names(param_results)) {
  result <- param_results[[result_name]]
  cat(sprintf("ε=%.1f, minPts=%d: %d clusters, %.1f%% noise\n",
              result$eps, result$min_pts, result$n_clusters, result$noise_percent))
}

# Choose parameters that give reasonable clustering 
# Let's choose parameters that give 2-5 clusters with reasonable noise
chosen_eps <- 0.7
chosen_min_pts <- 5

cat(paste("\nUsing chosen parameters: epsilon =", chosen_eps, "minPts =", chosen_min_pts, "\n"))

# Perform DBSCAN clustering with chosen parameters
dbscan_result <- dbscan(cosine_dist_dbscan, eps = chosen_eps, minPts = chosen_min_pts)
dbscan_clusters <- dbscan_result$cluster

cat("\nDBSCAN CLUSTERING RESULTS:\n")
print(table(dbscan_clusters))

# Calculate percentage of noise points
noise_percentage <- sum(dbscan_clusters == 0) / length(dbscan_clusters) * 100
cat(paste("Noise points (cluster 0):", round(noise_percentage, 2), "%\n"))

# Get top terms for DBSCAN clusters 
if(length(unique(dbscan_clusters)) > 1 && sum(dbscan_clusters != 0) > 0) {
  non_noise_indices <- which(dbscan_clusters != 0)
  non_noise_clusters <- dbscan_clusters[non_noise_indices]
  non_noise_data <- tfidf_sample_dbscan[non_noise_indices, ]
  
  unique_clusters <- sort(unique(non_noise_clusters))
  
  cluster_means <- t(sapply(unique_clusters, function(clust_id) {
    cluster_data <- non_noise_data[non_noise_clusters == clust_id, , drop = FALSE]
    if(nrow(cluster_data) > 1) {
      colMeans(cluster_data)
    } else {
      cluster_data  # Single document case
    }
  }))
  
  rownames(cluster_means) <- unique_clusters
  
  top_terms_dbscan <- get_top_terms(cluster_means)
  cat("\nTop terms for each DBSCAN cluster (excluding noise):\n")
  for (i in 1:length(top_terms_dbscan)) {
    cat(paste("Cluster", unique_clusters[i], ":", paste(top_terms_dbscan[[i]], collapse = ", "), "\n"))
  }
} else {
  cat("No meaningful clusters found besides noise.\n")
  
  # If only one cluster or all noise, try even smaller epsilon
  cat("Trying smaller epsilon (0.3) to get more clusters...\n")
  dbscan_result_retry <- dbscan(cosine_dist_dbscan, eps = 0.3, minPts = 5)
  dbscan_clusters_retry <- dbscan_result_retry$cluster
  
  cat("DBSCAN results with ε=0.3:\n")
  print(table(dbscan_clusters_retry))
  
  # Use the retry results if they're better
  if(length(unique(dbscan_clusters_retry)) > 1) {
    dbscan_clusters <- dbscan_clusters_retry
    noise_percentage <- sum(dbscan_clusters == 0) / length(dbscan_clusters) * 100
    cat(paste("Using retry results. Noise points:", round(noise_percentage, 2), "%\n"))
  }
}



# --------------------------- 
# PCA FOR CLUSTER VISUALIZATION
# ---------------------------
library(stats)
# PCA for K-means clustering
cat("\nPerforming PCA for K-means clustering visualization...\n")
pca_km <- prcomp(tfidf_sample_km, center = TRUE, scale. = TRUE)
# Extract the first two principal components for visualization
pca_km_data <- data.frame(PC1 = pca_km$x[, 1], 
                          PC2 = pca_km$x[, 2], 
                          Cluster = as.factor(kmeans_result$cluster))

# Calculate variance explained by PC1 and PC2
var_explained_km <- summary(pca_km)$importance[2, 1:2] * 100
cat("Variance explained by PC1 and PC2 for K-means:", 
    paste(round(var_explained_km, 2), "%"), "\n")
# Plot K-means clusters in 2D PCA space
kmeans_plot <- ggplot(pca_km_data, aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point(size = 2) +
  labs(title = "K-means Clusters in PCA Space",
       x = paste0("PC1 (", round(var_explained_km[1], 2), "%)"),
       y = paste0("PC2 (", round(var_explained_km[2], 2), "%)")) +
  theme_minimal() +
  scale_color_brewer(palette = "Set1")
print(kmeans_plot)

# PCA for hierarchical clustering
cat("\nPerforming PCA for hierarchical clustering visualization...\n")
pca_hc <- prcomp(tfidf_sample_hc, center = TRUE, scale. = TRUE)
# Extract the first two principal components for visualization
pca_hc_data <- data.frame(PC1 = pca_hc$x[, 1], 
                          PC2 = pca_hc$x[, 2], 
                          Cluster = as.factor(h_clusters))

# Calculate variance explained by PC1 and PC2
var_explained_hc <- summary(pca_hc)$importance[2, 1:2] * 100
cat("Variance explained by PC1 and PC2 for hierarchical clustering:", 
    paste(round(var_explained_hc, 2), "%"), "\n")

# Plot hierarchical clusters in 2D PCA space
hc_plot <- ggplot(pca_hc_data, aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point(size = 2) +
  labs(title = "Hierarchical Clusters in PCA Space",
       x = paste0("PC1 (", round(var_explained_hc[1], 2), "%)"),
       y = paste0("PC2 (", round(var_explained_hc[2], 2), "%)")) +
  theme_minimal() +
  scale_color_brewer(palette = "Set1")
print(hc_plot)

# PCA for DBSCAN clustering
cat("\nPerforming PCA for DBSCAN clustering visualization...\n")
pca_dbscan <- prcomp(tfidf_sample_dbscan, center = TRUE, scale. = TRUE)
pca_dbscan_data <- data.frame(PC1 = pca_dbscan$x[, 1], 
                              PC2 = pca_dbscan$x[, 2], 
                              Cluster = as.factor(dbscan_clusters))

var_explained_dbscan <- summary(pca_dbscan)$importance[2, 1:2] * 100
cat("Variance explained by PC1 and PC2 for DBSCAN:", 
    paste(round(var_explained_dbscan, 2), "%"), "\n")

dbscan_plot <- ggplot(pca_dbscan_data, aes(x = PC1, y = PC2, color = as.factor(Cluster))) +
  geom_point(size = 2) +
  labs(title = "DBSCAN Clusters in PCA Space",
       x = paste0("PC1 (", round(var_explained_dbscan[1], 2), "%)"),
       y = paste0("PC2 (", round(var_explained_dbscan[2], 2), "%)"),
       color = "Cluster") +
  theme_minimal() +
  scale_color_brewer(palette = "Set1") +
  guides(color = guide_legend(title = "Cluster\n(0 = Noise)"))
print(dbscan_plot)



# ---------------------------
# ADD CLUSTER ASSIGNMENTS TO ORIGINAL DATA
# ---------------------------

# Add k-means cluster assignments back to your original data
hotel_reviews_sample$kmeans_cluster <- NA
hotel_reviews_sample$kmeans_cluster[nonzero_rows][sample_indices_km] <- kmeans_result$cluster

# For hierarchical clustering
hotel_reviews_sample$hierarchical_cluster <- NA
hotel_reviews_sample$hierarchical_cluster[nonzero_rows][sample_indices_hc] <- h_clusters

# For dbscan clustering

hotel_reviews_sample$dbscan_cluster <- NA
hotel_reviews_sample$dbscan_cluster[nonzero_rows][sample_indices_dbscan] <- dbscan_clusters

# View the data with cluster assignments
cat("\nSAMPLE OF DATA WITH CLUSTER ASSIGNMENTS:\n")
sample_data <- head(hotel_reviews_sample[, c("Positive_Review", "Negative_Review", 
                                             "kmeans_cluster", "hierarchical_cluster", 
                                             "dbscan_cluster")])
print(sample_data)




# Print summary
cat("\n=== CLUSTERING SUMMARY ===\n")
cat(paste("Total documents processed:", nrow(tfidf_matrix), "\n"))
cat(paste("K-means sample size:", sample_size_km, "\n"))
cat(paste("Hierarchical clustering sample size:", sample_size_hc, "\n"))
cat(paste("DBSCAN sample size:", sample_size_dbscan, "\n"))
cat("K-means cluster sizes:\n")
print(table(kmeans_result$cluster))
cat("Hierarchical cluster sizes:\n")
print(table(h_clusters))

cat("DBSCAN cluster sizes (0 = noise):\n")
print(table(dbscan_clusters))
cat(paste("Noise points:", sum(dbscan_clusters == 0), "(", 
          round(noise_percentage, 2), "%)\n"))

# Create a data frame with the clustering summary - FIXED VERSION
summary_data <- data.frame(
  Category = c("Total Documents", "K-means Sample", "K-means Cluster 1", "K-means Cluster 2", 
               "K-means Cluster 3", "K-means Cluster 4", "Hierarchical Sample", 
               "Hierarchical Cluster 1", "Hierarchical Cluster 2", "Hierarchical Cluster 3", 
               "Hierarchical Cluster 4", "DBSCAN Sample", "DBSCAN Noise", "DBSCAN Cluster 1"),
  Count = c(nrow(tfidf_matrix), sample_size_km, 1663, 220, 79, 38, 
            sample_size_hc, 303, 16, 31, 150, sample_size_dbscan,
            13, 987),
  Type = c("Total", "K-means", "K-means", "K-means", "K-means", "K-means", 
           "Hierarchical", "Hierarchical", "Hierarchical", "Hierarchical", "Hierarchical",
           "DBSCAN", "DBSCAN", "DBSCAN")
)

# Create a grouped bar chart
p <- ggplot(summary_data, aes(x = Category, y = Count, fill = Type)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Clustering Summary of Hotel Reviews",
       x = "Categories",
       y = "Number of Documents") +
  scale_fill_manual(values = c("Total" = "#1f77b4", "K-means" = "#ff7f0e", 
                               "Hierarchical" = "#2ca02c", "DBSCAN" = "#d62728")) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_y_continuous(expand = c(0, 0), limits = c(0, max(summary_data$Count) * 1.1))

# Display the plot
print(p)


import collaborative

if __name__ == '__main__':
    collab_filtering = collaborative.CollaborativeBasedRecommender(collaborative.AlgorithmType.NMF)
    collab_filtering.knn.train_model()
    # collab_filtering.decomposition_algo.get_recommendations(15, 10)

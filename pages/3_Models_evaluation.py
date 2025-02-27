import streamlit as st
import collaborative
import statistics

st.set_page_config(layout="wide")
st.title('Models evaluation')
st.write('***')

collab_recommenders = collaborative.CollaborativeBasedRecommender()

st.header('Singular Value Decomposition')
collab_recommenders.decomposition_algo.set_algo(collaborative.AlgorithmType.SVD)
validate_results_svd, rmse_svd = collab_recommenders.decomposition_algo.validate_model(True)
svd_rmse, svd_mae, svd_mse, svd_fcp, svd_fit_time, svd_test_time = st.columns(6)

with svd_rmse:
    svd_rmse.metric('RMSE', round(rmse_svd, 3))
with svd_mae:
    svd_mae.metric('MAE', round(statistics.mean(validate_results_svd.get('test_mae')), 3))
with svd_mse:
    svd_mse.metric('MSE', round(statistics.mean(validate_results_svd.get('test_mse')), 3))
with svd_fcp:
    svd_fcp.metric('FCP', round(statistics.mean(validate_results_svd.get('test_fcp')), 3))
with svd_fit_time:
    svd_fit_time.metric('Fit time (mean)', round(statistics.mean(validate_results_svd.get('fit_time')), 4))
with svd_test_time:
    svd_test_time.metric('Test time (mean)', round(statistics.mean(validate_results_svd.get('test_time')), 4))

st.header('Non-negative Matrix Factorization')
collab_recommenders.decomposition_algo.set_algo(collaborative.AlgorithmType.NMF)
validate_results_nmf, rmse_nmf = collab_recommenders.decomposition_algo.validate_model(True)
nmf_rmse, nmf_mae, nmf_mse, nmf_fcp, nmf_fit_time, nmf_test_time = st.columns(6)

with nmf_rmse:
    nmf_rmse.metric('RMSE', round(rmse_nmf, 3))
with nmf_mae:
    nmf_mae.metric('MAE', round(statistics.mean(validate_results_nmf.get('test_mae')), 3))
with nmf_mse:
    nmf_mse.metric('MSE', round(statistics.mean(validate_results_nmf.get('test_mse')), 3))
with nmf_fcp:
    nmf_fcp.metric('FCP', round(statistics.mean(validate_results_nmf.get('test_fcp')), 3))
with nmf_fit_time:
    nmf_fit_time.metric('Fit time (mean)', round(statistics.mean(validate_results_nmf.get('fit_time')), 4))
with nmf_test_time:
    nmf_test_time.metric('Test time (mean)', round(statistics.mean(validate_results_nmf.get('test_time')), 4))

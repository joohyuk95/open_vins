
#include "IMMEstimator.h"
#include <iostream>
#include <fstream> 
#include <iomanip> 
#include <limits>

namespace ov_msckf
{   
    
    // IMMEstimator::IMMEstimator(std::string mu_str, std::string M_str) { 
    //     std::cout << "[IMMEstimator] Constructor called with mu = " << mu_str << ", M = " << M_str << std::endl;

    // }
    IMMEstimator::IMMEstimator() { 
            std::cout << "[IMMEstimator] Constructor called with mu =" << std::endl;
            num_models = 2;

            mu = Eigen::VectorXd(num_models);
            mu << 0.5, 0.5;

            M = Eigen::MatrixXd(num_models, num_models);
            M << 0.9, 0.1,
                0.1, 0.9;
            likelihoods = Eigen::VectorXd::Zero(num_models); 
    
            }
    void IMMEstimator::printDebugInfo() const {
    std::cout << "[IMMEstimator] printDebugInfo() called" << std::endl;
    }
    double IMMEstimator::likelihood(const Eigen::MatrixXd& S, const Eigen::MatrixXd& Sinv, const Eigen::VectorXd& res){
        // std::cout << "IMMEstimator called" <<std::endl;

        int n = res.size();
        // std::cout << "res size " << n << std::endl;
        Eigen::LLT<Eigen::MatrixXd> llt(S);

        if (llt.info() != Eigen::Success) {
        std::cerr << "LLT failed. S is not positive definite.\n";
        }
        // std::cout << "S_determinant\n" << S.determinant() << std::endl;

        Eigen::MatrixXd L = llt.matrixL();
        // std::cout << "res size " << L << std::endl;
        double log_det = 0.0;
        for (int i = 0; i < L.rows(); ++i){
            log_det += std::log(L(i, i));
        }
        log_det *= 2;
        // std::cout << "log det of S: " << log_det << std::endl;

        double mahabol = res.transpose() * Sinv * res;
        // std::cout << "mahalanobis distance: " << mahabol << std::endl;
        double log_likelihood = -0.5 * (n * std::log(2 * M_PI) + log_det + mahabol);
        // std::cout << "log likelihood: " << log_likelihood << std::endl;
        double normal_likelihood = std::exp(log_likelihood);
        // std::cout << "exp normal likelihood: " << normal_likelihood << std::endl;
 

        double direct_likelihood = 1.0 / std::sqrt(std::pow(2 * M_PI, res.size()) * S.determinant()) * std::exp(-0.5 * mahabol);
        std::cout << "direct likelihood \n" << direct_likelihood << std::endl;
        // double direct_likelihood = 1.0 / std::sqrt((2 * M_PI * S).determinant()) * std::exp(-0.5 * mahabol);
        // std::cout << "direct normal likelihood: " << direct_likelihood << std::endl;
        // double denom = std::sqrt((2 * M_PI * S).determinant());
        // std::cout << "denom \n" << denom << std::endl;
        // std::cout << "S diagonal:\n" << S.diagonal().transpose() << std::endl;
        // Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(S);
        // std::cout << "Min Eigenvalue: " << eig.eigenvalues().minCoeff() << std::endl;   
        // std::cout << "residual size: " << res.size() << std::endl;

        return direct_likelihood;
    }

    // double IMMEstimator::likelihood(const Eigen::MatrixXd& S, const Eigen::MatrixXd& Sinv, const Eigen::VectorXd& res) {
    //     int n = res.size();

    //     // Check if S is positive definite
    //     // Eigen::LLT<Eigen::MatrixXd> llt(S);
    //     // if (llt.info() != Eigen::Success) {
    //     //     std::cerr << "LLT failed. S is not positive definite.\n";
    //     //     return -1.0; // signal failure
    //     // }

    //     // Compute Mahalanobis distance
    //     double mahabol = res.transpose() * Sinv * res;

    //     // Compute determinant
    //     double detS = S.determinant();
    //     std::cout << "S_determinant: " << detS << std::endl;

    //     if (detS <= 0.0 || std::isinf(detS) || std::isnan(detS)) {
    //         std::cerr << "Invalid determinant. Cannot compute likelihood.\n";
    //         return std::numeric_limits<double>::quiet_NaN(); // signal failure
    //     }

    //     // Compute likelihood
    //     double norm_const = std::pow(2 * M_PI, n) * detS;
    //     double direct_likelihood = 1.0 / std::sqrt(norm_const) * std::exp(-0.5 * mahabol);
    //     std::cout << "direct likelihood: " << direct_likelihood << std::endl;

    //     return direct_likelihood;
    // }



    void IMMEstimator::saveModeProbHistoryCSV(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "[IMMEstimator] Could not open file to save mode probabilities: " << filename << std::endl;
            return;
        }

        // Optional: Write header
        file << "Timestamp";
        for (int i = 0; i < num_models; ++i) {
            file << ",Mode" << i;
        }
        file << "\n";

        // Write time and probabilities
        for (const auto& pair : mode_prob_history) {
            double timestamp = pair.first;
            const Eigen::VectorXd& mu_vec = pair.second;

            // Timestamp: fixed decimal (not scientific)
            file << std::fixed << std::setprecision(6) << timestamp;

            // Mode probabilities: scientific with higher precision
            for (int i = 0; i < mu_vec.size(); ++i) {
                file << "," << std::scientific << std::setprecision(6) << mu_vec[i];
            }

            file << "\n";
        }

        file.close();
        // std::cout << "[IMMEstimator] Mode probabilities saved to " << filename << std::endl;
    }


    void IMMEstimator::record_likelihood(int model_id, double likelihood){
        if (model_id >= 0 && model_id < likelihoods.size()){
            likelihoods[model_id] = likelihood;
        }
        else{
            std::cerr << "Invalid model_id in record_likelihood" << std::endl;
        }
        // std::cout << "likelihoods: " << likelihoods << std::endl;
    }

//     void IMMEstimator::record_likelihood(int model_id, double likelihood) {
//     if (model_id < 0 || model_id >= likelihoods.size()) {
//         std::cerr << "Invalid model_id in record_likelihood" << std::endl;
//         return;
//     }

//     // Threshold for considering likelihood as unchanged
//     const double threshold = 1e-10;

//     // Check if new likelihood is close to old likelihood for this model
//     double old_likelihood = likelihoods[model_id];
//     bool is_similar = std::abs(old_likelihood - likelihood) < threshold;

//     if (is_similar) {
//         // If likelihood is similar, keep likelihoods of other models unchanged
//         // (do nothing)
//         std::cout << "similar likelihood in \n " << model_id << std::endl; 
//         return;
        
//     } else {
//         // Update likelihood for the model_id only
//         likelihoods[model_id] = likelihood;

//         // Optionally: update other likelihoods if you compute them elsewhere
//         // or leave them unchanged if no new info is available
//     }
// }



    void IMMEstimator::updateModelProbabilities(double timestamp){
        
        Eigen::VectorXd normal_likelihoods = likelihoods;
        // std::cout << "normal likelihood: \n" << normal_likelihoods << std::endl;
        // std::cout << "mu: \n" << mu << std::endl;
        
        // Step 1: Validate likelihoods
        bool all_valid = true;
        for (int i = 0; i < normal_likelihoods.size(); ++i) {
            if (!std::isfinite(normal_likelihoods[i]) || normal_likelihoods[i] <= 0.0) {
                all_valid = false;
                break;
            }
        }

        if (!all_valid) {
            std::cerr << "[IMMEstimator] Skipping mode probability update due to invalid likelihoods.\n";
            return;  // Skip update and retain previous mu
        }


        Eigen::VectorXd unnormalized_mu(mu.size());
        for (int i = 0; i < mu.size(); i++){
            unnormalized_mu[i] = mu[i] * normal_likelihoods[i];
        }

        double sum = 0.0;
        for (int i = 0; i < mu.size(); i++){
            sum += unnormalized_mu[i];
        }
        // double sum_1 = unnormalized_mu.sum();
        // std::cout << "sum: " << sum << std::endl;
        // std::cout << "sum_1: " << sum_1 << std::endl;
        if (sum > 0) {
            mu = unnormalized_mu / sum;
        } else {
            std::cerr << "[IMMEstimator] Warning: sum of unnormalized probabilities is zero!" << std::endl;
            return;
        }
        std::cout << "Updated mode probabilities: \n" << mu.transpose() << std::endl;
        // std::vector<Eigen::VectorXd> mode_prob_history;
        mode_prob_history.emplace_back(timestamp, mu);

    }

    std::pair<Eigen::MatrixXd, Eigen::VectorXd> IMMEstimator::mixingProbabilities(){
        Eigen::MatrixXd mixing_probs(num_models, num_models);
    
        // Calculate the mixing probabilities
        // for (int i = 0; i < num_models; ++i){  // i: current model
        //     double denom = 0.0;
        //     // Calculate the denominator (sum over all models j)
        //     for (int j = 0; j < num_models; ++j){  // j: previous model
        //         denom += M(i, j) * mu(j);  // M(i, j) is the transition from j to i
        //     }
        //     // Calculate the mixing probabilities for each model transition
        //     for (int j = 0; j < num_models; ++j){  // j: previous model
        //         mixing_probs(i, j) = (M(i, j) * mu(j)) / denom;  // Mixing probability from j to i
        //     }
        // }
        for (int j = 0; j < num_models; ++j){
            double denom = 0.0;
            for (int i = 0; i < num_models; ++i){
                denom += mu(i) * M(i, j);
            }
            for (int i = 0; i < num_models; ++i){
                mixing_probs(j, i) = (mu(i) * M(i, j)) / denom;
            }
        }
    
        // std::cout << "Mixing probabilities: "
        //           << mixing_probs(0, 0) << ", "
        //           << mixing_probs(0, 1) << ", "
        //           << mixing_probs(1, 0) << ", "
        //           << mixing_probs(1, 1) << std::endl;
    
        return std::make_pair(mixing_probs, mu);
    }    
    

} // namespace ov_msckf